"""Head extraction regressions. Run: python -m unittest discover -s tests -v.

Model outputs are deterministic CPU stubs; cropping, PNG IO and PSD export are real.
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from psd_tools import PSDImage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
sys.path.insert(0, str(ROOT / 'inference' / 'scripts'))

from utils.cv import center_square_pad_resize, crop_layerdiff_head
from utils import inference_utils
import inference_psd_quantized as quantized


HEAD_TAGS = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite',
             'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']


def head_mask(rect=None, size=128, alpha=255):
    image = np.zeros((size, size, 4), dtype=np.uint8)
    if rect is not None:
        x, y, w, h = rect
        image[y:y + h, x:x + w] = [90, 100, 110, alpha]
    return image


class HeadCropTests(unittest.TestCase):
    def crop(self, source_shape, mask):
        source = np.arange(np.prod(source_shape), dtype=np.uint8).reshape(source_shape)
        _, pad_size, pad_pos = center_square_pad_resize(
            source, mask.shape[0], return_pad_info=True)
        return crop_layerdiff_head(source, mask, pad_size[0] / mask.shape[0], pad_pos)

    def test_empty_and_subthreshold_masks_skip_with_diagnostics(self):
        for mask in [head_mask(), head_mask((20, 20, 30, 30), alpha=15)]:
            with self.subTest(alpha=mask[..., -1].max()), redirect_stdout(io.StringIO()) as log:
                self.assertIsNone(self.crop((128, 128, 4), mask))
                self.assertIn('Skipping head pass', log.getvalue())
                self.assertIn('mask bbox=', log.getvalue())
                self.assertIn('source size=128x128', log.getvalue())

    def test_masks_entirely_in_padding_skip(self):
        cases = [
            ((128, 64, 4), (0, 40, 16, 24)),
            ((128, 64, 4), (112, 40, 16, 24)),
            ((64, 128, 4), (40, 0, 24, 16)),
            ((64, 128, 4), (40, 112, 24, 16)),
        ]
        for shape, rect in cases:
            with self.subTest(shape=shape, rect=rect), redirect_stdout(io.StringIO()):
                self.assertIsNone(self.crop(shape, head_mask(rect)))

    def test_mask_rounding_to_zero_source_pixels_skips(self):
        for rect in [(60, 40, 1, 24), (40, 60, 24, 1)]:
            with self.subTest(rect=rect), redirect_stdout(io.StringIO()):
                self.assertIsNone(self.crop((16, 16, 4), head_mask(rect)))

    def test_partial_overlap_is_clipped_to_each_source_edge(self):
        cases = [
            ((128, 64, 4), (20, 40, 24, 24), (0, 36, 12, 68)),
            ((128, 64, 4), (84, 40, 24, 24), (52, 36, 64, 68)),
            ((64, 128, 4), (40, 20, 24, 24), (36, 0, 68, 12)),
            ((64, 128, 4), (40, 84, 24, 24), (36, 52, 68, 64)),
        ]
        for shape, rect, expected in cases:
            with self.subTest(shape=shape, rect=rect):
                crop, bounds = self.crop(shape, head_mask(rect))
                self.assertEqual(bounds, expected)
                x1, y1, x2, y2 = expected
                self.assertEqual(crop.shape, (y2 - y1, x2 - x1, 4))

    def test_valid_crop_keeps_context_and_source_pixels(self):
        source = np.arange(128 * 128 * 4, dtype=np.uint8).reshape(128, 128, 4)
        crop, bounds = crop_layerdiff_head(source, head_mask((48, 20, 24, 24)), 1, (0, 0))
        self.assertEqual(bounds, (44, 16, 76, 48))
        np.testing.assert_array_equal(crop, source[16:48, 44:76])

    def test_crop_accounts_for_scale_and_padding(self):
        crop, bounds = self.crop((160, 80, 4), head_mask((20, 8, 8, 12), size=64))
        self.assertEqual(bounds, (6, 14, 34, 56))
        self.assertEqual(crop.shape, (42, 28, 4))


class StubLayerdiff:
    def __init__(self, mask):
        self.mask = mask
        self.unet = SimpleNamespace(device='cpu', get_tag_version=lambda: 'v3')
        self.calls = []
        self.body_images = {}

    def cache_tag_embeds(self):
        pass

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, *, prompt, fullpage, group_index, **kwargs):
        self.calls.append(group_index)
        images = []
        for tag in prompt:
            image = np.zeros_like(fullpage)
            if group_index == 1:
                image[:] = [120, 130, 140, 255]
            elif tag == 'head':
                image = self.mask.copy()
            elif tag == 'back hair':
                image[8:56, 20:108] = [30, 40, 50, 255]
            elif tag == 'topwear':
                image[48:104, 24:104] = [60, 70, 80, 255]
            if group_index == 0:
                self.body_images[tag] = image.copy()
            images.append(image)
        return SimpleNamespace(images=images)


class StubMarigold:
    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, *, img_list, **kwargs):
        count = len(img_list)
        h, w = img_list[0].shape[:2]
        return SimpleNamespace(depth_tensor=torch.full((count, h, w), 0.5))


class HeadPipelineTests(unittest.TestCase):
    def run_layers(self, mode, pipeline, source, output):
        if mode == 'standard':
            with patch.object(inference_utils, 'layerdiff_pipeline', pipeline):
                inference_utils.apply_layerdiff(str(source), 'unused', save_dir=str(output),
                                               resolution=128, num_inference_steps=1)
        else:
            # Blockswap imports and uses this same run_layerdiff function.
            quantized.run_layerdiff(pipeline, str(source), str(output), seed=42,
                                    num_inference_steps=1, resolution=128)

    def read_image(self, path):
        with Image.open(path) as image:
            return np.array(image)

    def test_invalid_head_skips_second_pass_and_clears_previous_head_layers(self):
        cases = [((128, 128), head_mask()),
                 ((128, 128), head_mask((48, 20, 24, 24), alpha=15)),
                 ((128, 64), head_mask((0, 40, 16, 24))),
                 ((16, 16), head_mask((60, 40, 1, 24)))]
        for mode in ['standard', 'quantized']:
            for shape, mask in cases:
                with self.subTest(mode=mode, shape=shape, alpha=mask[..., -1].max()), \
                        tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log:
                    output = Path(tmp)
                    source = output / 'character.png'
                    Image.new('RGBA', shape[::-1], (200, 180, 160, 255)).save(source)
                    saved = output / 'character'
                    saved.mkdir()
                    for tag in HEAD_TAGS + ['src_head']:
                        Image.new('RGBA', (128, 128), (255, 0, 0, 255)).save(saved / f'{tag}.png')

                    pipeline = StubLayerdiff(mask)
                    self.run_layers(mode, pipeline, source, output)

                    self.assertEqual(pipeline.calls, [0])
                    self.assertIn('Skipping head pass', log.getvalue())
                    for tag in HEAD_TAGS + ['src_head']:
                        image = self.read_image(saved / f'{tag}.png')
                        self.assertEqual(image.shape, (128, 128, 4))
                        self.assertFalse(image.any(), tag)
                    for tag, image in pipeline.body_images.items():
                        np.testing.assert_array_equal(self.read_image(saved / f'{tag}.png'), image)

    def test_valid_head_runs_second_pass_and_keeps_output_alignment(self):
        for mode in ['standard', 'quantized']:
            for shape in [(128, 128), (128, 64), (64, 128)]:
                with self.subTest(mode=mode, shape=shape), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp)
                    source = output / 'character.png'
                    Image.new('RGBA', shape[::-1], (200, 180, 160, 255)).save(source)
                    rect = (20, 48, 24, 24) if shape[0] == 64 else (48, 20, 24, 24)
                    pipeline = StubLayerdiff(head_mask(rect))
                    self.run_layers(mode, pipeline, source, output)
                    self.assertEqual(pipeline.calls, [0, 1])
                    expected = np.zeros((128, 128, 4), dtype=np.uint8)
                    if shape[0] == 64:
                        expected[44:76, 16:48] = [120, 130, 140, 255]
                    else:
                        expected[16:48, 44:76] = [120, 130, 140, 255]
                    for tag in HEAD_TAGS:
                        np.testing.assert_array_equal(
                            self.read_image(output / 'character' / f'{tag}.png'), expected)

    def test_no_head_completes_depth_and_psd_export_with_body_and_hair(self):
        for mode in ['standard', 'quantized']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, \
                    redirect_stdout(io.StringIO()):
                output = Path(tmp)
                source = output / 'character.png'
                Image.new('RGBA', (128, 128), (200, 180, 160, 255)).save(source)
                self.run_layers(mode, StubLayerdiff(head_mask()), source, output)
                depth = StubMarigold()
                if mode == 'standard':
                    with patch.object(inference_utils, 'marigold_pipeline', depth):
                        inference_utils.apply_marigold(str(source), 'unused', save_dir=str(output),
                                                      resolution=64, num_inference_steps=1)
                else:
                    quantized.run_marigold(depth, str(source), str(output), seed=42, resolution_depth=64)
                inference_utils.further_extr(str(output / 'character'), rotate=False,
                                            save_to_psd=True, tblr_split=False)
                for name in ['character.psd', 'character_depth.psd']:
                    psd = PSDImage.open(output / name)
                    self.assertEqual(psd.size, (128, 128))
                    self.assertEqual({layer.name for layer in psd}, {'topwear', 'back hair'})


if __name__ == '__main__':
    unittest.main()
