from heapq import heappop, heappush
from itertools import combinations

import cv2
import numpy as np

from utils.cv import bbox_intersection


_GARMENTS = {'topwear', 'bottomwear', 'footwear'}


def _overlap_votes(garment, limb, source):
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
    skin = limb['img'][top - ly:bottom - ly, left - lx:right - lx]
    original = source[top:bottom, left:right]
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
    fragment = dict(garment, img=img, depth=depth, xyxy=[left + x, top + y, left + x + w, top + y + h],
                    tag=f'{tag}-{side}', source_tag=tag, depth_median=float(np.median(depth[img[..., 3] > 10])) / 255)
    fragment.pop('layer_name', None)
    fragment.pop('stack_depth', None)
    return fragment


def _distance(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


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


def _fragment_alpha(img, front, mixed):
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
    front_alpha = np.rint(alpha * coverage).astype(np.uint8)

    # Solve source-over for the original alpha, so the shared edge never opens
    # a gap when the two fragments are shown together.
    remaining = 255 - front_alpha
    rear = np.zeros(alpha.shape, dtype=np.float32)
    np.divide((alpha - front_alpha) * 255.0, remaining, out=rear, where=remaining > 0)
    rear_alpha = np.rint(rear).astype(np.uint8)
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
    split = bool(front_matches and back_matches)
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
    front_alpha, rear_alpha, fill = _fragment_alpha(img, cut, mixed)
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
    by_tag = {part['tag']: part for part in parts if part.get('visible', True)}
    arms = [by_tag[tag] for tag in ('handwear', 'handwear-l', 'handwear-r') if tag in by_tag]
    legs = [by_tag['legwear']] if 'legwear' in by_tag else []
    overlaps = {'topwear': arms + legs, 'bottomwear': arms + legs, 'footwear': legs}
    result = []
    order = []
    for part in parts:
        limbs = overlaps.get(part['tag'])
        if limbs and part.get('visible', True):
            fragments, constraints = _split_garment(part, limbs, source)
            result.extend(fragments)
            order.extend(constraints)
        else:
            result.append(part)
    order.extend(_garment_order(result, source))
    return _apply_order(result, order) if order else result
