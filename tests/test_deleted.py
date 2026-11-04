"""Real PNG/PSD IO with deterministic CPU model stubs; no downloads required."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))

from utils import deleted, inference_utils
from utils.cv import batch_save_masks
from test_head_crop import StubLayerdiff, StubMarigold, head_mask


class DeletedPixelsTests(unittest.TestCase):
    def test_unions_masks_and_preserves_exact_rgba_including_soft_edges(self):
        original = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
        original[0, 0, 3] = 0
        original[1, 4, 3] = 1
        original[2, 5, 3] = 127
        masks = np.zeros((2, 4, 6), dtype=np.uint8)
        masks[0, :, :2] = 1
        masks[1, :2, 1:3] = 255
        union = deleted.union_masks(masks, original.shape[:2])
        before = original.copy()
        result = deleted.deleted_pixels(original, union)
        missing = (original[..., 3] > 0) & ~np.any(masks, axis=0)
        np.testing.assert_array_equal(result[missing], original[missing])
        self.assertFalse(result[~missing].any())
        np.testing.assert_array_equal(original, before)
        for y, x in [(1, 4), (2, 5)]:
            np.testing.assert_array_equal(result[y, x], original[y, x])

    def test_empty_and_complete_segmentation(self):
        original = np.full((3, 5, 4), 255, dtype=np.uint8)
        np.testing.assert_array_equal(deleted.deleted_pixels(original, deleted.union_masks([], (3, 5))), original)
        self.assertFalse(deleted.deleted_pixels(original, np.ones((3, 5), dtype=bool)).any())
        original[..., 3] = 0
        self.assertFalse(deleted.deleted_pixels(original, np.zeros((3, 5), dtype=bool)).any())

    def test_rejects_misaligned_masks(self):
        with self.assertRaisesRegex(ValueError, 'mask shape'):
            deleted.union_masks([np.ones((5, 3))], (3, 5))

    def test_semanticsam_uses_original_rgb_and_unions_foreground_logits(self):
        import torch
        from unittest.mock import Mock
        from modules.semanticsam import SemanticSam
        original = np.full((5, 7, 4), [11, 22, 33, 127], dtype=np.uint8)
        logits = torch.full((19, 5, 7), -1.0)
        logits[0, :, :2] = 1
        logits[18, :3, :] = 1
        logits[1, 4, 6] = 0  # Logit zero is not foreground.
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        model.inference.return_value = [logits]
        with patch('utils.torch_utils.init_model_from_pretrained', return_value=model) as load:
            union = deleted.segment_original(original)
        np.testing.assert_array_equal(model.inference.call_args.args[0], original[..., :3])
        np.testing.assert_array_equal(union, np.any(logits.numpy() > 0, axis=0))
        self.assertEqual(load.call_args.kwargs['module_cls'], SemanticSam)
        self.assertEqual(load.call_args.kwargs['model_args'], {'class_num': 19})
        model.eval.assert_called_once()

    def test_snapshot_accepts_original_masks_and_rejects_images_without_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'source.png'
            Image.new('RGBA', (7, 5), (11, 22, 33, 127)).save(source)
            masks = np.zeros((2, 5, 7), dtype=bool)
            masks[0, :, :2] = True
            masks[1, :3, :] = True
            mask_path = directory / 'masks.json'
            batch_save_masks(masks, str(mask_path))
            with patch.object(deleted, 'segment_original') as model:
                deleted.prepare_deleted(source, directory, 64, mask_path)
                model.assert_not_called()
            pixels = deleted.load_deleted(directory, (64, 64))
            expected = np.zeros((5, 7, 4), dtype=np.uint8)
            expected[3:, 2:] = [11, 22, 33, 127]
            np.testing.assert_array_equal(pixels, expected)
            with self.assertRaisesRegex(ValueError, 'inference canvas'):
                deleted.load_deleted(directory, (128, 128))
            Image.new('RGB', (7, 5)).save(source)
            with self.assertRaisesRegex(ValueError, 'alpha channel'):
                deleted.prepare_deleted(source, directory, 64)


class DeletedExportTests(unittest.TestCase):
    def test_empty_deleted_still_exports_as_hidden_full_canvas_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'empty.psd'
            inference_utils.dump_parts_psd({}, (8, 8), str(target),
                                          deleted=np.zeros((5, 7, 4), dtype=np.uint8))
            psd = PSDImage.open(target)
            self.assertEqual(psd.size, (7, 5))
            self.assertEqual([layer.name for layer in psd], ['Deleted'])
            self.assertEqual(psd[0].bbox, (0, 0, 7, 5))
            self.assertFalse(psd[0].visible)
            self.assertFalse(np.array(psd[0].topil()).any())

    def test_restore_source_coordinates_for_wide_tall_and_scaled_canvases(self):
        for source_shape, frame_size in [((4, 8), (8, 8)), ((8, 4), (8, 8)),
                                         ((8, 16), (8, 8)), ((5, 8), (8, 8))]:
            with self.subTest(source_shape=source_shape, frame_size=frame_size):
                h, w = source_shape
                side = max(h, w)
                part = {'tag': 'part', 'img': np.full((2, 2, 4), 255, dtype=np.uint8),
                        'depth': np.full((2, 2), 128, dtype=np.uint8), 'xyxy': [3, 3, 5, 5]}
                restored = deleted.restore_parts_to_source([part], frame_size, source_shape)[0]
                x1, y1, x2, y2 = restored['xyxy']
                full = np.zeros((h, w, 4), dtype=np.uint8)
                full[y1:y2, x1:x2] = restored['img']
                # The center of the generated part must map to the source center,
                # with the original integer padding restored, even for odd margins.
                cy, cx = side // 2 - (side-h)//2, side // 2 - (side-w)//2
                self.assertEqual(full[cy, cx, 3], 255)
                self.assertEqual(restored['depth'].shape, restored['img'].shape[:2])
                self.assertEqual(part['xyxy'], [3, 3, 5, 5])

    def test_hidden_full_canvas_layer_survives_psd_roundtrip_and_reexport(self):
        for size in [(7, 11), (11, 7)]:
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                h, w = size
                original = np.full((h, w, 4), [31, 47, 93, 127], dtype=np.uint8)
                original[2, 3] = [29, 35, 127, 1]
                residual = deleted.deleted_pixels(original, np.zeros(size, dtype=bool))
                parts = {'part': {'tag': 'part', 'img': np.full((8, 8, 4), 255, dtype=np.uint8),
                                  'depth': np.full((8, 8), 128, dtype=np.uint8),
                                  'xyxy': [0, 0, 8, 8], 'depth_median': .5}}
                target = str(Path(tmp) / 'result.psd')
                inference_utils.dump_parts_psd(parts, (8, 8), target, deleted=residual)
                for name in ['result.psd', 'result_depth.psd']:
                    psd = PSDImage.open(Path(tmp) / name)
                    self.assertEqual(psd.size, (w, h))
                    layer = next(layer for layer in psd if layer.name == 'Deleted')
                    self.assertFalse(layer.visible)
                    self.assertEqual(layer.opacity, 255)
                    self.assertEqual(layer.bbox, (0, 0, w, h))
                psd = PSDImage.open(target)
                layer = next(layer for layer in psd if layer.name == 'Deleted')
                np.testing.assert_array_equal(np.array(layer.topil()), residual)
                # A hidden Deleted layer must not change the visible composite.
                np.testing.assert_array_equal(np.array(psd.composite().convert('RGBA')),
                                              np.full((h, w, 4), 255, dtype=np.uint8))
                loaded = inference_utils.psd2partdicts(target)
                self.assertFalse(loaded['parts']['Deleted']['visible'])
                second = str(Path(tmp) / 'second.psd')
                inference_utils.dump_parts_psd(loaded['parts'], loaded['frame_size'], second)
                layer = next(layer for layer in PSDImage.open(second) if layer.name == 'Deleted')
                self.assertFalse(layer.visible)
                np.testing.assert_array_equal(np.array(layer.topil()), residual)

    def test_cli_snapshots_masks_before_generation_even_if_inpainting_covers_missing_pixels(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            source = directory / 'character.png'
            original = np.full((141, 87, 4), [211, 113, 57, 255], dtype=np.uint8)
            original[:5, :, 3] = 0
            original[81, 41] = [13, 79, 201, 1]
            Image.fromarray(original).save(source)
            union = np.ones(original.shape[:2], dtype=bool)
            union[80, 40] = union[81, 41] = False
            pipeline = StubLayerdiff(head_mask())
            def segment(image):
                self.assertEqual(pipeline.calls, [])
                np.testing.assert_array_equal(image, original)
                return union
            with patch.object(deleted, 'segment_original', side_effect=segment) as segmentation, \
                    patch.object(inference_utils, 'layerdiff_pipeline', pipeline), \
                    patch.object(inference_utils, 'marigold_pipeline', StubMarigold()), \
                    patch.object(sys, 'argv', ['inference_psd.py', '--srcp', str(source),
                                              '--save_dir', tmp, '--resolution', '128',
                                              '--resolution_depth', '64', '--save_to_psd', '--save_deleted']):
                runpy.run_path(str(ROOT / 'inference/scripts/inference_psd.py'), run_name='__main__')
            segmentation.assert_called_once()
            psd = PSDImage.open(directory / 'character.psd')
            layer = next(layer for layer in psd if layer.name == 'Deleted')
            self.assertEqual(psd.size, (87, 141))
            expected = deleted.deleted_pixels(original, union)
            np.testing.assert_array_equal(np.array(layer.topil()), expected)
            self.assertFalse(layer.visible)
            # The visible generated topwear covers this lost source pixel. Its
            # final alpha must never be used to decide what goes into Deleted.
            topwear = next(layer for layer in psd if layer.name == 'topwear')
            x1, y1, _, _ = topwear.bbox
            self.assertGreater(np.array(topwear.topil())[80-y1, 40-x1, 3], 0)
            info = json.loads((directory / 'character.psd.json').read_text())
            self.assertEqual(info['frame_size'], [141, 87])
            self.assertFalse(info['parts']['Deleted']['visible'])

    def test_directory_segmentation_finishes_before_cached_generative_models_load(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            sources = directory / 'inputs'
            sources.mkdir()
            for name in ['one', 'two']:
                Image.new('RGBA', (128, 128), (211, 113, 57, 255)).save(sources / f'{name}.png')
            events = []
            def segment(image):
                events.append('segment')
                return np.ones(image.shape[:2], dtype=bool)
            def generate(*args, **kwargs):
                self.assertEqual(events[:2], ['segment', 'segment'])
                events.append('generate')
            with patch.object(deleted, 'segment_original', side_effect=segment), \
                    patch.object(inference_utils, 'apply_layerdiff', side_effect=generate), \
                    patch.object(inference_utils, 'apply_marigold'), \
                    patch.object(inference_utils, 'further_extr'), \
                    patch.object(sys, 'argv', ['inference_psd.py', '--srcp', str(sources),
                                              '--save_dir', str(directory / 'output'),
                                              '--save_to_psd', '--save_deleted']):
                runpy.run_path(str(ROOT / 'inference/scripts/inference_psd.py'), run_name='__main__')
            self.assertEqual(events, ['segment', 'segment', 'generate', 'generate'])


if __name__ == '__main__':
    unittest.main()
