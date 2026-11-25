from heapq import heappop, heappush
from itertools import combinations
import warnings

import cv2
import numpy as np

from utils.arm_pose import arm_regions, complete_hidden_bone, estimate_arm_joints
from utils.cv import bbox_intersection


_GARMENTS = {'topwear', 'bottomwear', 'footwear'}
_ARM_SECTION_RANK = {'upperarm': 0, 'forearm': 1, 'hand': 2}


def _layer_overlap(garment, limb, source):
    h, w = source.shape[:2]
    box = bbox_intersection(garment['xyxy'], limb['xyxy'])
    box = bbox_intersection(box, (0, 0, w, h)) if box is not None else None
    if box is None:
        return None
    left, top, right, bottom = box

    gx, gy = garment['xyxy'][:2]
    lx, ly = limb['xyxy'][:2]
    region = np.s_[top - gy:bottom - gy, left - gx:right - gx]
    cloth = garment['img'][region]
    local = np.s_[top - ly:bottom - ly, left - lx:right - lx]
    skin = limb['img'][local]
    original = source[top:bottom, left:right]
    return region, cloth, skin, original


def _overlap_votes(garment, limb, source, *, thin_edges=False):
    layers = _layer_overlap(garment, limb, source)
    if layers is None:
        return None
    region, cloth, skin, original = layers
    overlap = (original[..., 3] >= 224) & (cloth[..., 3] >= 192) & (skin[..., 3] >= 192)
    area = np.count_nonzero(overlap)
    if area < 12:
        return None

    original_rgb = original[overlap, :3].astype(np.int16)
    cloth_error = np.abs(cloth[overlap, :3] - original_rgb).mean(axis=1)
    limb_error = np.abs(skin[overlap, :3] - original_rgb).mean(axis=1)
    seeds = []
    for winner, loser in [(cloth_error, limb_error), (limb_error, cloth_error)]:
        # Require a close color match and a coherent patch of it. Boundary pixels
        # and tiny islands are poor evidence of which image was visible.
        votes = np.zeros(overlap.shape, dtype=np.uint8)
        votes[overlap] = (winner <= 40) & (loser - winner >= 12)
        if not thin_edges:
            votes = cv2.morphologyEx(votes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(votes, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= max(12, area * 0.015)
        keep[0] = False
        seeds.append(keep[labels])
    return region, seeds, area


def _place(garment, front_depths, back_depths):
    depth = float(garment['depth_median'])
    if front_depths and depth >= min(front_depths):
        return dict(garment, stack_depth=float(np.nextafter(min(front_depths), -np.inf)))
    if back_depths and depth <= max(back_depths):
        return dict(garment, stack_depth=float(np.nextafter(max(back_depths), np.inf)))
    return garment


def _clothing_fragment(garment, alpha, side, fill=None):
    keep = alpha > 0
    x, y, w, h = cv2.boundingRect(keep.astype(np.uint8))
    region = np.s_[y:y + h, x:x + w]
    img = garment['img'][region].copy()
    depth = garment['depth'][region].copy()
    img[..., 3] = alpha[region]
    if fill is not None:
        padding, colors = fill
        img[..., :3][padding[region]] = colors
    img[~keep[region]] = 0
    depth[~keep[region]] = 255
    left, top = garment['xyxy'][:2]
    tag = garment['tag']
    opaque = img[..., 3] > 10
    median = float(np.median(depth[opaque])) / 255 if opaque.any() else garment['depth_median']
    fragment = dict(garment, img=img, depth=depth, xyxy=[left + x, top + y, left + x + w, top + y + h],
                    tag=f'{tag}-{side}', source_tag=tag, depth_median=median)
    fragment.pop('layer_name', None)
    fragment.pop('stack_depth', None)
    return fragment


def _distance(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def _source_matte(garment, limbs, source, cut, mixed):
    # Estimate cloth coverage only close to the cut and only where the source
    # pixel can be explained by a blend of the two generated colors.
    seam = mixed & (_distance(cut) <= 4) & (_distance(~cut) <= 4)
    coverage = np.zeros(cut.shape, dtype=np.float32)
    confidence = np.zeros_like(coverage)
    for limb in limbs:
        layers = _layer_overlap(garment, limb, source)
        if layers is None:
            continue
        region, cloth, skin, original = layers
        eligible = seam[region] & (original[..., 3] >= 240) & (cloth[..., 3] >= 240) & (skin[..., 3] >= 240)
        if not eligible.any():
            continue
        background = skin[eligible, :3].astype(np.float32)
        colors = cloth[eligible, :3] - background
        observed = original[eligible, :3] - background
        contrast2 = np.sum(colors * colors, axis=1)
        fraction = np.clip(np.sum(observed * colors, axis=1) / np.maximum(contrast2, 1), 0, 1)
        error = np.sqrt(np.mean((observed - fraction[:, None] * colors) ** 2, axis=1))
        weight = np.clip((12 - error) / 8, 0, 1) * np.clip((np.sqrt(contrast2) - 30) / 25, 0, 1)
        better = weight > confidence[region][eligible]
        update = eligible.copy()
        update[eligible] = better
        coverage[region][update] = fraction[better]
        confidence[region][update] = weight[better]
    return coverage, confidence


def _apply_source_matte(coverage, contour, matte):
    source_coverage, confidence = matte
    if not confidence.any():
        return coverage
    h, w = confidence.shape
    padded_values = np.pad(source_coverage, 1)
    padded_confidence = np.pad(confidence, 1)
    padded_contour = np.pad(contour, 1, mode='edge')
    neighbors = np.zeros_like(confidence)
    weighted = source_coverage * confidence
    for y, x in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        region = np.s_[y:y + h, x:x + w]
        # Compare points along the contour rather than averaging through it.
        tangent = np.clip(1 - np.abs(padded_contour[region] - contour) / 0.65, 0, 1)
        weight = padded_confidence[region] * tangent
        neighbors += weight
        weighted += padded_values[region] * weight
    guided = weighted / np.maximum(confidence + neighbors, 1e-6)
    blend = confidence * np.clip(neighbors, 0, 1)
    return coverage * (1 - blend) + guided * blend


def _follow_cloth_edges(img, front, back, mixed, nearest_front):
    # GrabCut can follow trim and folds beyond the part that overlaps the limb.
    # On flat art, keep the spatial estimate because color adds no guidance.
    front_rgb = np.median(img[front, :3], axis=0)
    back_rgb = np.median(img[back, :3], axis=0)
    if np.abs(front_rgb - back_rgb).mean() < 5:
        return nearest_front

    rgb = img[..., :3].copy()
    silhouette = img[..., 3] > 0
    rgb[~silhouette] = 0
    labels = np.where(nearest_front, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[~silhouette] = cv2.GC_BGD
    labels[front] = cv2.GC_FGD
    labels[back] = cv2.GC_BGD
    cv2.grabCut(rgb, labels, None, None, None, 2, cv2.GC_INIT_WITH_MASK)
    return np.where(mixed, (labels & 1) != 0, nearest_front)


def _fragment_alpha(img, front, mixed, *, extend=True, matte=None):
    alpha = img[..., 3]
    front = front & (alpha > 0)
    back = ~front & (alpha > 0)
    front_distance, back_distance = _distance(front), _distance(back)
    signed = np.where(front, back_distance - 0.5, 0.5 - front_distance)
    smoothed = cv2.GaussianBlur(signed, (5, 5), 1.1)
    contour = signed + np.clip(smoothed - signed, -0.65, 0.65)
    dx = cv2.Sobel(contour, cv2.CV_32F, 1, 0, ksize=1, scale=0.5)
    dy = cv2.Sobel(contour, cv2.CV_32F, 0, 1, ksize=1, scale=0.5)
    # The contour covers more of a pixel diagonally than along a straight row.
    # Limit its displacement so narrow hems and folds keep their shape.
    width = np.maximum(1, np.abs(dx) + np.abs(dy))
    seam = mixed & (front_distance <= 2) & (back_distance <= 2)
    coverage = np.where(seam, np.clip(0.5 + contour / width, 0, 1), front)
    if matte is not None:
        coverage = _apply_source_matte(coverage, contour, matte)
    front_alpha = np.rint(alpha * coverage).astype(np.uint8)

    # Solve source-over for the original alpha, so the shared edge never opens
    # a gap when the two fragments are shown together.
    remaining = 255 - front_alpha
    rear = np.zeros(alpha.shape, dtype=np.float32)
    np.divide((alpha - front_alpha) * 255.0, remaining, out=rear, where=remaining > 0)
    rear_alpha = np.rint(rear).astype(np.uint8)
    if not extend:
        return front_alpha, rear_alpha, None
    padding = mixed & (front_alpha == 255) & (back_distance <= 3)
    rear_alpha[padding] = 255
    if not padding.any():
        return front_alpha, rear_alpha, None

    # Extend the lining under the opaque front edge using nearby lining colors.
    _, labels = cv2.distanceTransformWithLabels((~back).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    colors = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
    colors[labels[back]] = img[back, :3]
    return front_alpha, rear_alpha, (padding, colors[labels[padding]])


def _split_arms(parts, source):
    arms = [part for part in parts if part.get('visible', True)
            and part['tag'] in ('handwear', 'handwear-l', 'handwear-r')]
    if not arms:
        return parts, [], {}
    try:
        joints = estimate_arm_joints(source, arms)
    except (OSError, cv2.error) as exc:
        warnings.warn(f'Arm pose model could not run: {exc}', stacklevel=2)
        return parts, [], {}
    if not joints:
        return parts, [], {}

    arm_tags = {part['tag'] for part in arms}
    result, order = [], []
    completions = {}
    for part in parts:
        if part['tag'] not in arm_tags:
            result.append(part)
            continue
        remaining = part['img'][..., 3].copy()
        cut = False
        for side, component, distal, hand, bones in arm_regions(part, joints):
            img = part['img'].copy()
            img[..., 3][~component] = 0
            distal_alpha, upper_alpha, _ = _fragment_alpha(img, distal, component, extend=False)
            img[..., 3] = distal_alpha
            hand_alpha, forearm_alpha, _ = _fragment_alpha(img, hand, component, extend=False)
            shoulder, elbow, wrist = bones
            arm = dict(part, tag=f'handwear-{side}')
            upper = _clothing_fragment(arm, upper_alpha, 'upperarm')
            forearm = _clothing_fragment(arm, forearm_alpha, 'forearm')
            palm = _clothing_fragment(arm, hand_alpha, 'hand')
            completions[upper['tag']] = (arm, upper_alpha, distal_alpha, shoulder, elbow)
            completions[forearm['tag']] = (arm, forearm_alpha, hand_alpha, elbow, wrist)
            result.extend((upper, forearm, palm))
            order.extend(((upper['tag'], forearm['tag'], (3, 0)), (forearm['tag'], palm['tag'], (3, 0))))
            remaining[component] = 0
            cut = True
        if not cut:
            result.append(part)
        elif remaining.any():
            remainder = _clothing_fragment(part, remaining, 'remaining')
            result.append(dict(remainder, tag=part['tag'], depth_median=part['depth_median']))
    return result, order, completions


def _complete_arms(parts, completions):
    if not completions:
        return parts
    ordered = sorted(parts, key=_stack_depth, reverse=True)
    for i, part in enumerate(ordered):
        if part['tag'] not in completions:
            continue
        arm, observed, cover, start, end = completions[part['tag']]
        cover = cover.copy()
        # Let the sleeve hide the anatomical outline as well. Any added pixels
        # then remain covered in the assembled pose, without tracing fingers.
        ax, ay = arm['xyxy'][:2]
        for garment in ordered[i + 1:]:
            if garment.get('source_tag', garment['tag']) not in _GARMENTS or not garment.get('visible', True):
                continue
            box = bbox_intersection(arm['xyxy'], garment['xyxy'])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            gx, gy = garment['xyxy'][:2]
            area = cover[y1 - ay:y2 - ay, x1 - ax:x2 - ax]
            np.maximum(area, garment['img'][y1 - gy:y2 - gy, x1 - gx:x2 - gx, 3], out=area)
        completed, fill = complete_hidden_bone(arm['img'], observed, cover, start, end)
        if fill is not None:
            depth = arm['depth'].copy()
            depth[fill[0]] = round(part['depth_median'] * 255)
            updated = _clothing_fragment(dict(arm, depth=depth), completed, part['tag'].rsplit('-', 1)[1], fill)
            updated['depth_median'] = part['depth_median']
            if 'stack_depth' in part:
                updated['stack_depth'] = part['stack_depth']
            ordered[i] = updated
    return ordered


def _between_arm_sections(front_matches, back_matches):
    # If the garment can sit between already split bones, keep it in one piece.
    bounds = []
    for matches in (front_matches, back_matches):
        ranks = {}
        for part, _ in matches:
            arm, _, section = part['tag'].rpartition('-')
            if arm not in ('handwear-l', 'handwear-r') or section not in _ARM_SECTION_RANK:
                return False
            ranks.setdefault(arm, []).append(_ARM_SECTION_RANK[section])
        bounds.append(ranks)
    front, back = bounds
    return all(max(front[arm]) < min(back[arm]) for arm in front.keys() & back.keys())


def _split_garment(garment, limbs, source):
    front = np.zeros(garment['img'].shape[:2], dtype=bool)
    back = np.zeros_like(front)
    front_matches, back_matches = [], []
    for limb in limbs:
        evidence = _overlap_votes(garment, limb, source)
        if evidence is None:
            continue
        region, (front_votes, back_votes), area = evidence
        front_count, back_count = np.count_nonzero(front_votes), np.count_nonzero(back_votes)
        # Moving an entire layer on one-sided evidence needs broader support.
        if front_count and (back_count or front_count >= 0.15 * area):
            front[region] |= front_votes
            front_matches.append((limb, region, front_votes))
        if back_count and (front_count or back_count >= 0.15 * area):
            back[region] |= back_votes
            back_matches.append((limb, region, back_votes))

    if front_matches and back_matches:
        conflict = front & back
        front &= ~conflict
        back &= ~conflict
        if not front.any() or not back.any():
            return [garment], []

    placements = []
    for mask, matches in ((front, front_matches), (back, back_matches)):
        support = [(limb, np.count_nonzero(mask[region] & votes)) for limb, region, votes in matches]
        placements.append([(limb, count) for limb, count in support if count])
    front_matches, back_matches = placements
    front_depths = [float(limb['depth_median']) for limb, _ in front_matches]
    back_depths = [float(limb['depth_median']) for limb, _ in back_matches]
    tag = garment['tag']
    split = bool(front_matches and back_matches) and not _between_arm_sections(front_matches, back_matches)
    front_tag = f'{tag}-front' if split else tag
    back_tag = f'{tag}-back' if split else tag
    order = [(limb['tag'], front_tag, (2, count)) for limb, count in front_matches]
    order.extend((back_tag, limb['tag'], (2, count)) for limb, count in back_matches)
    if not split:
        return [_place(garment, front_depths, back_depths)], order

    img = garment['img']
    _, components = cv2.connectedComponents((img[..., 3] > 0).astype(np.uint8))
    has_front = np.isin(components, np.unique(components[front]))
    has_back = np.isin(components, np.unique(components[back]))
    default_front = garment['depth_median'] < np.median(front_depths + back_depths)
    mixed = has_front & has_back
    nearest_front = (has_front & ~has_back) | (mixed & (_distance(front) <= _distance(back)))
    nearest_front |= ~has_front & ~has_back & default_front
    cut = _follow_cloth_edges(img, front, back, mixed, nearest_front) if mixed.any() else nearest_front
    matte = _source_matte(garment, limbs, source, cut, mixed) if mixed.any() else None
    front_alpha, rear_alpha, fill = _fragment_alpha(img, cut, mixed, matte=matte)
    order.append((back_tag, front_tag, (3, 0)))
    return [
        _place(_clothing_fragment(garment, front_alpha, 'front'), front_depths, []),
        _place(_clothing_fragment(garment, rear_alpha, 'back', fill), [], back_depths),
    ], order


def _garment_order(parts, source):
    garments = [(part, part.get('source_tag', part['tag'])) for part in parts if part.get('visible', True)]
    garments = [(part, tag) for part, tag in garments if tag in _GARMENTS]
    for (first, first_tag), (second, second_tag) in combinations(garments, 2):
        if first_tag == second_tag:
            continue
        evidence = _overlap_votes(first, second, source)
        if evidence is None:
            continue
        _, votes, area = evidence
        first_count, second_count = map(np.count_nonzero, votes)
        if first_count >= 0.15 * area and not second_count:
            yield second['tag'], first['tag'], (1, first_count)
        elif second_count >= 0.15 * area and not first_count:
            yield first['tag'], second['tag'], (1, second_count)


def _headwear_order(by_tag, source):
    accessory = by_tag.get('headwear')
    if accessory is None:
        return
    for tag in ('face', 'front hair', 'back hair', 'hair', 'hairf', 'hairb'):
        if tag not in by_tag:
            continue
        evidence = _overlap_votes(accessory, by_tag[tag], source, thin_edges=True)
        if evidence is None:
            continue
        _, votes, area = evidence
        front, back = map(np.count_nonzero, votes)
        if front >= 0.12 * area and front > 9 * back:
            yield tag, 'headwear', (1, front)
        elif back >= 0.12 * area and back > 9 * front:
            yield 'headwear', tag, (1, back)


def _stack_depth(part):
    return float(part.get('stack_depth', part['depth_median']))


def _apply_order(parts, constraints):
    # Add reliable relations first. Drop a weaker relation if accepting it
    # would create a cycle in the order of whole layers.
    preferred = sorted(parts, key=_stack_depth, reverse=True)
    ranks = {part['tag']: i for i, part in enumerate(preferred)}
    followers = [set() for _ in preferred]
    successors = [set() for _ in preferred]
    incoming = [0] * len(preferred)
    for behind, ahead, weight in sorted(constraints, key=lambda edge: edge[2], reverse=True):
        start, end = ranks[behind], ranks[ahead]
        if start == end or start in successors[end] or end in successors[start]:
            continue
        followers[start].add(end)
        incoming[end] += 1
        after_end = {end} | successors[end]
        for i, reachable in enumerate(successors):
            if i == start or start in reachable:
                reachable.update(after_end)

    ready = [i for i, count in enumerate(incoming) if not count]
    order = []
    while ready:
        i = heappop(ready)
        order.append(i)
        for j in followers[i]:
            incoming[j] -= 1
            if not incoming[j]:
                heappush(ready, j)

    if order == list(range(len(preferred))):
        return parts
    # Reuse the depth slots so the PSD and its metadata retain the same order
    # on a later export without the original PNG.
    result = []
    for slot, index in enumerate(order):
        part = preferred[index]
        depth = _stack_depth(preferred[slot])
        result.append(dict(part, stack_depth=depth) if _stack_depth(part) != depth else part)
    return result


def split_clothing_layers(parts, source):
    """Fit garment stacking to visible overlaps on the aligned original image."""
    parts, order, completions = _split_arms(parts, source)
    by_tag = {part['tag']: part for part in parts if part.get('visible', True)}
    arms = [part for tag, part in by_tag.items() if tag.split('-', 1)[0] == 'handwear']
    legs = [by_tag['legwear']] if 'legwear' in by_tag else []
    overlaps = {'topwear': arms + legs, 'bottomwear': arms + legs, 'footwear': legs}
    result = []
    for part in parts:
        limbs = overlaps.get(part['tag'])
        if limbs and part.get('visible', True):
            fragments, constraints = _split_garment(part, limbs, source)
            result.extend(fragments)
            order.extend(constraints)
        else:
            result.append(part)
    order.extend(_garment_order(result, source))
    order.extend(_headwear_order(by_tag, source))
    if order:
        result = _apply_order(result, order)
    return _complete_arms(result, completions)
