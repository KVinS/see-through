"""Preserve original pixels missed by segmentation, before LayerDiff inpainting."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def union_masks(masks, shape):
    """Combine binary foreground masks in the original image coordinates."""
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        mask = np.asarray(mask)
        if mask.shape != tuple(shape):
            raise ValueError(f'Segmentation mask shape {mask.shape} != source shape {shape}')
        union |= mask > 0
    return union


def deleted_pixels(original, segmentation_union):
    """Copy RGBA bytes verbatim, including partially transparent edge pixels."""
    if original.ndim != 3 or original.shape[2] != 4:
        raise ValueError('Deleted requires an RGBA original')
    if segmentation_union.shape != original.shape[:2]:
        raise ValueError('Segmentation and original must have the same dimensions')
    missing = (original[..., 3] > 0) & ~segmentation_union.astype(bool)
    deleted = np.zeros_like(original)
    deleted[missing] = original[missing]
    return deleted


def segment_original(original):
    """Use the upstream 19-part SemanticSAM on the original, never generated layers.

    LayerDiff V3 generates completed parts directly; it has no separate visible
    segmentation output. This optional pass follows bodypartseg_sam.ipynb.
    Release the model before loading LayerDiff to avoid overlapping GPU weights.
    """
    import gc
    import torch
    from modules.semanticsam import SemanticSam
    from utils.torch_utils import init_model_from_pretrained

    model = None
    try:
        model = init_model_from_pretrained(
            pretrained_model_name_or_path='24yearsold/l2d_sam_iter2',
            weights_name='checkpoint-18000.pt', module_cls=SemanticSam,
            download_from_hf=True, model_args=dict(class_num=19),
        ).eval().to(device='cuda')
        with torch.inference_mode():
            predictions = model.inference(original[..., :3])[0]
            union = (predictions > 0).any(dim=0).cpu().numpy()
            del predictions
        return union
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def prepare_deleted(srcp, saved, resolution, masks_path=None):
    """Snapshot segmentation and residual RGBA before any generative processing."""
    with Image.open(srcp) as image:
        if 'A' not in image.getbands() and 'transparency' not in image.info:
            raise ValueError('--save_deleted requires an original image with an alpha channel')
        original = np.array(image.convert('RGBA'))
    h, w = original.shape[:2]
    if masks_path is not None:
        from utils.cv import batch_load_masks
        union = union_masks(batch_load_masks(str(masks_path)), (h, w))
    elif np.any(original[..., 3]):
        print('Segmenting original for Deleted (SemanticSAM, before LayerDiff)...', flush=True)
        union = union_masks([segment_original(original)], (h, w))
    else:
        union = np.zeros((h, w), dtype=bool)
    deleted = deleted_pixels(original, union)
    saved = Path(saved)
    saved.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8) * 255).save(saved / 'segmentation_union.png')
    Image.fromarray(deleted).save(saved / 'deleted.png')
    metadata = {'source_size': [h, w], 'inference_size': [resolution, resolution]}
    (saved / 'deleted.json').write_text(json.dumps(metadata), encoding='utf-8')


def load_deleted(saved, frame_size):
    saved = Path(saved)
    metadata = json.loads((saved / 'deleted.json').read_text(encoding='utf-8'))
    if tuple(metadata['inference_size']) != tuple(frame_size):
        raise ValueError('Deleted snapshot does not match the inference canvas')
    with Image.open(saved / 'deleted.png') as image:
        pixels = np.array(image.convert('RGBA'))
    if pixels.shape[:2] != tuple(metadata['source_size']):
        raise ValueError('Deleted snapshot does not match the original canvas')
    return pixels


def restore_source_canvas(canvas, source_size):
    h, w = source_size
    side = max(h, w)
    px, py = (side - w) // 2, (side - h) // 2
    resized = cv2.resize(canvas, (side, side), interpolation=cv2.INTER_LINEAR)
    return resized[py:py+h, px:px+w]


def restore_parts_to_source(parts, frame_size, source_size):
    """Undo the centered square padding and resize for generated parts only.

    Deleted itself is never resized. Clip cropped parts to the inference canvas
    before the inverse transform; some optimized parts extend into its padding.
    """
    ih, iw = frame_size
    restored = []
    for part in parts:
        item = dict(part)
        x1, y1, x2, y2 = part.get('xyxy', [0, 0, iw, ih])
        left, top, right, bottom = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
        arrays = {}
        for key, fill in [('img', 0), ('depth', 255)]:
            array = part[key]
            canvas = np.full((ih, iw) + array.shape[2:], fill, dtype=array.dtype)
            if left < right and top < bottom:
                canvas[top:bottom, left:right] = array[top-y1:bottom-y1, left-x1:right-x1]
            arrays[key] = restore_source_canvas(canvas, source_size)
        bx, by, bw, bh = cv2.boundingRect((arrays['img'][..., 3] > 0).astype(np.uint8))
        if bw == 0 or bh == 0:
            bx = by = 0
            bw = bh = 1
        for key, array in arrays.items():
            item[key] = array[by:by+bh, bx:bx+bw].copy()
        item.pop('mask', None)
        item['xyxy'] = [bx, by, bx+bw, by+bh]
        restored.append(item)
    return restored
