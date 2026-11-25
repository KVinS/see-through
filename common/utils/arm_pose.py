import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


_INPUT_SIZE = (192, 256)
_ARM_JOINTS = {'l': [5, 7, 9], 'r': [6, 8, 10]}
_DEFAULT_MODEL = Path(__file__).resolve().parents[3] / 'models_cache' / 'arm_pose' / 'rtmpose-t.onnx'


@lru_cache(maxsize=1)
def _load_model(path):
    return cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), dtype=np.uint8))


def estimate_arm_joints(source, arms):
    """Run RTMPose locally on the original character, in PSD coordinates."""
    path = Path(os.environ.get('SUGOI_ARM_POSE_MODEL', _DEFAULT_MODEL))
    if not path.is_file():
        return {}
    x, y, w, h = cv2.boundingRect((source[..., 3] >= 128).astype(np.uint8))
    if w == 0 or h == 0:
        return {}
    bottom = min(y + h, max(arm['xyxy'][3] for arm in arms) + 0.18 * h)
    height = max(bottom - y, 0.45 * h)
    model_w, model_h = _INPUT_SIZE
    scale = 1 / (1.25 * max(w / model_w, height / model_h))
    offset = np.array([model_w / 2 - scale * (x + w / 2), model_h / 2 - scale * (y + height / 2)])
    affine = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]])
    alpha = source[..., 3:4].astype(np.float32) / 255
    rgb = source[..., :3] * alpha + 230 * (1 - alpha)
    crop = cv2.warpAffine(rgb, affine, _INPUT_SIZE, borderValue=(230, 230, 230))
    crop = (crop - np.float32([123.675, 116.28, 103.53])) / np.float32([58.395, 57.12, 57.375])
    net = _load_model(path)
    net.setInput(np.ascontiguousarray(crop.transpose(2, 0, 1)[None]))
    simcc_x, simcc_y = net.forward(['simcc_x', 'simcc_y'])
    joints = np.stack((simcc_x[0].argmax(axis=-1), simcc_y[0].argmax(axis=-1)), axis=-1) / 2
    scores = np.minimum(simcc_x[0].max(axis=-1), simcc_y[0].max(axis=-1))
    return {side: (joints[indices] - offset) / scale for side, indices in _ARM_JOINTS.items()
            if np.all(scores[indices] >= 0.40)}


def _segment_distance(x, y, start, end):
    delta = end - start
    length2 = delta @ delta
    t = np.clip(((x - start[0]) * delta[0] + (y - start[1]) * delta[1]) / length2, 0, 1)
    return (x - start[0] - t * delta[0]) ** 2 + (y - start[1] - t * delta[1]) ** 2


def _trace_crease(gray, opaque):
    width = opaque.shape[1]
    x = np.arange(width)
    rows = np.flatnonzero(opaque.any(axis=1))
    if rows.size < 8:
        return None
    left = opaque.argmax(axis=1)
    span = width - 1 - opaque[:, ::-1].argmax(axis=1) - left
    fraction = (x - left[:, None]) / np.maximum(span[:, None], 1)
    candidates = opaque & (fraction >= 0.09) & (fraction <= 0.72) & (span[:, None] >= 8)
    prior = 8 * (fraction - 0.32) ** 2
    ink = np.maximum(cv2.blur(gray, (9, 1)).astype(np.float32) - gray, 0)
    cost = np.where(candidates, gray / 40 - ink / 22 + prior, 40 + np.abs(fraction - 0.32) * span[:, None])
    steps = np.arange(-6, 7)
    neighbors = np.clip(x + steps[:, None], 0, width - 1)
    previous = cost[rows[0]]
    trace = []
    for row in rows[1:]:
        options = previous[neighbors] + 0.20 * np.abs(steps[:, None])
        best = options.argmin(axis=0)
        trace.append(neighbors[best, x])
        previous = cost[row] + options[best, x]
    path = [previous.argmin()]
    for links in reversed(trace):
        path.append(links[path[-1]])
    path = np.array(path[::-1])
    # A fold with no consistent ink or shading boundary is ambiguous.
    contrast = np.max(np.where(candidates, gray, 0), axis=1)[rows].astype(int) - gray[rows, path]
    if np.count_nonzero((contrast >= 20) & candidates[rows, path]) < 0.22 * rows.size:
        return None
    cut = np.interp(np.arange(opaque.shape[0]), rows, path)
    return cv2.GaussianBlur(cut[:, None], (1, 7), 1.4)[:, 0], left, span


def _folded_forearm(img, component, shoulder, elbow, wrist):
    # Rectify along the forearm. The crease separating two overlapping bones
    # becomes a continuous path along the rows of this image.
    bx, by, w, h = cv2.boundingRect(component.astype(np.uint8))
    rect = np.s_[by:by + h, bx:bx + w]
    elbow = elbow - (bx, by)
    shoulder = shoulder - (bx, by)
    wrist = wrist - (bx, by)
    length = np.linalg.norm(wrist - elbow)
    axis = (wrist - elbow) / length
    across = np.array([-axis[1], axis[0]])
    if (shoulder - elbow) @ across > 0:
        across *= -1
    basis = np.stack((across, axis))
    corners = (np.array([[0, 0], [w, 0], [0, h], [w, h]]) - elbow) @ basis.T
    low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
    affine = np.column_stack((basis, -basis @ elbow - low))
    size = tuple((high - low + 1).astype(int))
    warped = cv2.warpAffine(img[rect], affine, size)
    mask = cv2.warpAffine(component[rect].astype(np.uint8), affine, size, flags=cv2.INTER_NEAREST) > 0
    gray = cv2.cvtColor(warped[..., :3], cv2.COLOR_RGB2GRAY)
    trace = _trace_crease(gray, mask & (warped[..., 3] >= 128))
    if trace is None:
        return None
    cut, left, span = trace
    axial = np.arange(size[1]) + low[1]
    taper = np.clip((axial + 0.13 * length) / (0.30 * length), 0, 1)
    cut = np.minimum(cut, left + taper * span)
    cut = cv2.GaussianBlur(cut[:, None], (1, 7), 1.4)[:, 0]
    # Project the continuous curve back before rasterizing at export size.
    y, x = np.ogrid[:h, :w]
    column = across[0] * x + across[1] * y + affine[0, 2]
    row = axis[0] * x + axis[1] * y + affine[1, 2]
    boundary = np.interp(row, np.arange(size[1]), cut)
    distal = component[rect] & ((column >= boundary - 1.75) | (row + low[1] <= -0.13 * length))
    _, labels = cv2.connectedComponents((distal & (img[rect][..., 3] > 10)).astype(np.uint8))
    wx, wy = np.rint(wrist).astype(int)
    region = labels[wy, wx]
    if not region:
        return None
    attached = cv2.dilate((labels == region).astype(np.uint8), np.ones((5, 5), np.uint8))
    result = np.zeros(component.shape, dtype=bool)
    result[rect] = distal & (attached > 0)
    return result & component


def arm_regions(part, joints):
    """Match projected bones to distinct arm silhouettes, then cut at the joints."""
    alpha = part['img'][..., 3]
    h, w = alpha.shape
    left, top = part['xyxy'][:2]
    _, labels = cv2.connectedComponents((alpha > 10).astype(np.uint8))
    matches = {}
    for side, track in joints.items():
        track = track - (left, top)
        xy = np.rint(track).astype(int)
        if np.any(xy < 0) or np.any(xy >= (w, h)):
            continue
        ids = labels[xy[:, 1], xy[:, 0]]
        lengths = np.linalg.norm(np.diff(track, axis=0), axis=1)
        if ids[0] and np.all(ids == ids[0]) and np.all(lengths >= 8):
            matches.setdefault(ids[0], []).append((side, track))

    y, x = np.ogrid[:h, :w]
    for label, matches_for_component in matches.items():
        if len(matches_for_component) != 1:
            continue
        side, (shoulder, elbow, wrist) = matches_for_component[0]
        core = labels == label
        # Include the antialiased pixels immediately outside the component.
        component = core | ((labels == 0) & (alpha > 0) & (cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
        upper_axis, forearm_axis = shoulder - elbow, wrist - elbow
        hand = component & ((x - wrist[0]) * forearm_axis[0] + (y - wrist[1]) * forearm_axis[1] >= 0)
        angle_cos = (upper_axis @ forearm_axis) / (np.linalg.norm(upper_axis) * np.linalg.norm(forearm_axis))
        if angle_cos >= 0.80:
            distal = _folded_forearm(part['img'], component, shoulder, elbow, wrist)
            if distal is None:
                continue
        else:
            palm = wrist + 0.35 * forearm_axis
            distances = np.stack((_segment_distance(x, y, shoulder, elbow),
                                  _segment_distance(x, y, elbow, wrist),
                                  _segment_distance(x, y, wrist, palm)))
            labels_by_bone = distances.argmin(axis=0)
            distal = component & (labels_by_bone != 0)
        palm_mask = (core & distal & hand).astype(np.uint8)
        if angle_cos >= 0.80:
            # A thin bridge at a folded shoulder can attach its silhouette to
            # the palm. Find the palm core before restoring its soft edges.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            palm_mask = cv2.morphologyEx(palm_mask, cv2.MORPH_OPEN, kernel)
        count, fingers, stats, _ = cv2.connectedComponentsWithStats(palm_mask)
        if count > 2 or (count == 2 and angle_cos >= 0.80):
            main = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
            attached = cv2.dilate((fingers == main).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            distal &= ~hand | attached
            hand &= attached
        counts = np.array([np.count_nonzero(core & mask) for mask in (~distal, distal & ~hand, distal & hand)])
        if np.all(counts >= max(12, 0.03 * core.sum())):
            yield side, component, distal, hand, (shoulder, elbow, wrist)


def complete_hidden_bone(img, visible_alpha, cover_alpha, start, end):
    """Continue a tapered bone silhouette inside its opaque occluder."""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    length = np.linalg.norm(end - start)
    axis = (end - start) / length
    along = (x - start[0]) * axis[0] + (y - start[1]) * axis[1]
    across = (y - start[1]) * axis[0] - (x - start[0]) * axis[1]
    visible = visible_alpha >= 240
    supported = visible & (along >= 0.10 * length) & (along <= 0.85 * length)
    if not supported.any():
        return visible_alpha, None
    offsets = across[supported]
    width = np.clip(np.percentile(np.abs(offsets), 90), 0.08 * length, 0.27 * length)
    # On a folded arm the pose axis can sit towards the occluding forearm.
    # Let the visible upper-arm profile pull the outline back towards itself.
    bias = np.clip(np.median(offsets), -0.28 * width, 0.28 * width)
    radius = width * (1.05 - 0.40 * np.clip(along / length, 0, 1))
    delta = along - np.clip(along, 0, length)
    edge = radius - np.hypot(across - bias, delta)
    shape = np.clip(edge + 0.5, 0, 1)

    # Use occlusion as clearance for an independently shaped contour. Small
    # islands, fingers and hard edges of the front layer cannot become its rim.
    covered = cover_alpha == 255
    available = (covered | (visible_alpha > 10)).astype(np.uint8)
    span = 2 * max(1, round(0.16 * width)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span))
    available = cv2.morphologyEx(available, cv2.MORPH_OPEN, kernel)
    clearance = cv2.distanceTransform(available, cv2.DIST_L2, 5)
    margin = max(1.5, 0.09 * width)
    support = np.clip((clearance - 0.5) / margin, 0, 1)
    coverage = np.rint(255 * shape * support).astype(np.uint8)
    # Close pinholes where the fitted contour meets the observed silhouette.
    # Apply only under the cover, so a sparse rim cannot copy the hand shape.
    fitted = np.maximum(visible_alpha, np.where(covered, coverage, 0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coverage = cv2.morphologyEx(fitted, cv2.MORPH_CLOSE, kernel)
    proposed = np.maximum(visible_alpha, np.where(covered, coverage, 0))
    _, holes, stats, _ = cv2.connectedComponentsWithStats((proposed < 185).astype(np.uint8))
    small = stats[:, cv2.CC_STAT_AREA] <= max(4, round(0.8 * width))
    small[0] = False
    small[np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]]] = False
    pinholes = small[holes] & (cover_alpha >= 246)
    coverage[pinholes] = 255
    added = (covered | pinholes) & (coverage > visible_alpha)
    if not added.any():
        return visible_alpha, None

    # Borrow color from inside the visible segment so the fold's ink is not
    # extended across the fill. Add a soft outline only on the new perimeter.
    interior = cv2.erode(visible.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    donors = interior if interior.any() else visible
    _, labels = cv2.distanceTransformWithLabels((~donors).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    colors = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
    colors[labels[donors]] = img[donors, :3]
    rgb = cv2.GaussianBlur(colors[labels], (7, 7), 1.6)[added]
    completed = np.where(added, coverage, visible_alpha)
    perimeter = cv2.distanceTransform((completed >= 128).astype(np.uint8), cv2.DIST_L2, 5)
    shade = 0.70 + 0.30 * np.clip((perimeter[added] - 0.5) / 1.6, 0, 1)
    fill = np.rint(rgb * shade[:, None]).astype(np.uint8)
    return completed, (added, fill)
