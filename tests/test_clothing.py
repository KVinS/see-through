import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))

from utils.clothing import _apply_order, _fragment_alpha, split_clothing_layers
from utils.deleted import restore_parts_to_source, restore_source_canvas
from utils.inference_utils import dump_parts_psd, further_extr, psd2partdicts


def make_parts(garment_tag, limb_tag):
    garment = np.zeros((80, 80, 4), dtype=np.uint8)
    limb = np.zeros_like(garment)
    garment[10:70, 8:62] = (25, 75, 180, 255)
    garment[40, 8] = (25, 75, 180, 5)
    limb[5:72, 28:46] = (220, 150, 110, 255)
    source = np.array(Image.alpha_composite(Image.fromarray(garment), Image.fromarray(limb)))
    source[10:38, 28:46] = garment[10:38, 28:46]
    parts = [{'tag': tag, 'img': img, 'xyxy': [0, 0, 80, 80],
              'depth': np.full(img.shape[:2], depth, dtype=np.uint8), 'depth_median': depth / 255}
             for tag, img, depth in [(garment_tag, garment, 190), (limb_tag, limb, 60)]]
    return parts, source


def layer_on_canvas(layer, size):
    canvas = Image.new('RGBA', size)
    canvas.paste(layer.topil(), (layer.left, layer.top))
    return np.asarray(canvas)


def over(front, back):
    return np.asarray(Image.alpha_composite(Image.fromarray(back), Image.fromarray(front)))


class ClothingExportTests(unittest.TestCase):
    def test_both_overlap_directions_export_as_two_layers(self):
        for garment_tag, limb_tag in [('topwear', 'handwear'), ('bottomwear', 'legwear'), ('footwear', 'legwear')]:
            with self.subTest(garment=garment_tag), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / 'character'
                directory.mkdir()
                parts, source = make_parts(garment_tag, limb_tag)
                Image.fromarray(source).save(directory / 'src_img.png')
                for part in parts:
                    tag = part['tag']
                    Image.fromarray(part['img']).save(directory / f'{tag}.png')
                    Image.fromarray(part['depth']).save(directory / f'{tag}_depth.png')
                (directory / 'info.json').write_text(json.dumps({'parts': {part['tag']: {} for part in parts}}))

                further_extr(str(directory), rotate=False, save_to_psd=True, tblr_split=False)
                target = directory.with_suffix('.psd')
                psd = PSDImage.open(target)
                depths = PSDImage.open(directory.with_name('character_depth.psd'))
                names = [f'{garment_tag}-back', limb_tag, f'{garment_tag}-front']
                self.assertEqual([layer.name for layer in psd], names)
                self.assertEqual([layer.name for layer in depths], names)
                self.assertTrue(depths[0].has_mask())
                self.assertTrue(depths[2].has_mask())
                back = layer_on_canvas(psd[0], psd.size)
                front = layer_on_canvas(psd[2], psd.size)
                self.assertTrue(np.any((front[..., 3] == 255) & (back[..., 3] == 255)))
                np.testing.assert_array_equal(over(front, back), parts[0]['img'])
                self.assertFalse(psd[0].has_mask())
                self.assertFalse(psd[2].has_mask())
                np.testing.assert_array_equal(layer_on_canvas(psd[1], psd.size), parts[1]['img'])
                composite = np.array(psd.composite(force=True))
                opaque = source[..., 3] == 255
                np.testing.assert_allclose(composite[opaque], source[opaque], atol=1, rtol=0)

                loaded = psd2partdicts(str(target))
                reexport = str(directory.with_name('reexport.psd'))
                dump_parts_psd(loaded['parts'], loaded['frame_size'], reexport)
                self.assertEqual([layer.name for layer in PSDImage.open(reexport)], names)
                reexport_depths = PSDImage.open(directory.with_name('reexport_depth.psd'))
                for before, after in zip(depths, reexport_depths):
                    np.testing.assert_array_equal(np.asarray(before.topil()), np.asarray(after.topil()))

    def test_slit_follows_lining_below_leg_and_underskirt_stays_behind(self):
        size = (100, 100)
        y, x = np.indices(size)
        silhouette = (y >= 5) & (y < 95) & (x >= 8) & (x < 92)
        opening = silhouette & (y >= 12) & (np.abs(x - 50) < 3 + y * 0.2)
        outer = np.zeros((*size, 4), dtype=np.uint8)
        inner = np.zeros_like(outer)
        leg = np.zeros_like(outer)
        outer[silhouette] = (248, 236, 218, 255)
        outer[opening] = (154, 166, 186, 255)
        inner[12:92, 15:85] = (86, 92, 108, 255)
        leg[10:60, 37:63] = (241, 158, 118, 255)
        source = outer.copy()
        visible_leg = opening & (leg[..., 3] > 0)
        source[visible_leg] = leg[visible_leg]
        parts = [{'tag': tag, 'img': img, 'xyxy': [0, 0, 100, 100], 'depth_median': median,
                  'depth': np.full(size, depth, dtype=np.uint8)}
                 for tag, img, median, depth in [('topwear', outer, 0.30, 200),
                                                  ('bottomwear', inner, 0.40, 102),
                                                  ('legwear', leg, 0.50, 128)]]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'dress.psd')
            dump_parts_psd({}, size, path, part_dict_list=parts, source=source)
            psd = PSDImage.open(path)
            self.assertEqual([layer.name for layer in psd], ['bottomwear', 'topwear-back', 'legwear', 'topwear-front'])
            front = layer_on_canvas(psd[3], psd.size)
            back = layer_on_canvas(psd[1], psd.size)
            self.assertEqual(front[84, 50, 3], 0)
            self.assertEqual(front[84, 20, 3], 255)
            self.assertEqual(back[84, 50, 3], 255)
            self.assertEqual(back[84, 20, 3], 0)
            under_edge = (y >= 65) & ~opening & (front[..., 3] == 255) & (back[..., 3] == 255)
            self.assertTrue(np.any(under_edge))
            np.testing.assert_array_equal(back[under_edge, :3], np.tile([154, 166, 186], (under_edge.sum(), 1)))

    def test_one_sided_support_reorders_without_splitting_and_similar_colors_stay_ambiguous(self):
        parts, source = make_parts('bottomwear', 'legwear')
        source[10:70, 28:46] = parts[0]['img'][10:70, 28:46]
        garment, limb = split_clothing_layers(parts, source)
        self.assertEqual(garment['tag'], 'bottomwear')
        self.assertLess(garment['stack_depth'], limb['depth_median'])
        np.testing.assert_array_equal(garment['img'], parts[0]['img'])

        parts[1]['img'][..., :3] = parts[0]['img'][20, 30, :3]
        garment, limb = split_clothing_layers(parts, source)
        self.assertIs(garment, parts[0])
        self.assertIs(limb, parts[1])

    def test_garment_overlap_can_correct_order_even_without_a_limb_layer(self):
        parts, source = make_parts('footwear', 'bottomwear')
        source[10:70, 28:46] = parts[0]['img'][10:70, 28:46]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'clothing.psd')
            dump_parts_psd({}, source.shape[:2], path, part_dict_list=parts, source=source)
            self.assertEqual([layer.name for layer in PSDImage.open(path)], ['bottomwear', 'footwear'])

    def test_order_resolves_a_cycle_in_favor_of_the_limb_overlap(self):
        tags = ['lining', 'shoe', 'leg', 'front']
        parts = [{'tag': tag, 'depth_median': 1 - i / 4} for i, tag in enumerate(tags)]
        constraints = [('lining', 'leg', (2, 100)), ('leg', 'front', (2, 100)),
                       ('front', 'lining', (1, 200)), ('leg', 'shoe', (1, 80))]
        parts = _apply_order(parts, constraints)
        actual = sorted(parts, key=lambda part: part.get('stack_depth', part['depth_median']), reverse=True)
        self.assertEqual([part['tag'] for part in actual], ['lining', 'leg', 'shoe', 'front'])

    def test_mask_antialiases_diagonal_without_blurring_a_straight_cut(self):
        size = (60, 60)
        y, x = np.indices(size)
        img = np.full((*size, 4), 255, dtype=np.uint8)
        mixed = np.ones(size, dtype=bool)
        straight, _, _ = _fragment_alpha(img, x < 30, mixed)
        np.testing.assert_array_equal(straight, np.where(x < 30, 255, 0))

        front, back, _ = _fragment_alpha(img, x + y < 60, mixed)
        fractional = (front > 0) & (front < 255)
        self.assertTrue(np.all(fractional[12:48][(x + y == 59)[12:48]]))
        self.assertFalse(np.any(fractional[np.abs(x + y - 59.5) >= 2]))
        np.testing.assert_array_equal((front + (255 - front) * (back / 255)), img[..., 3])

    def test_source_size_export_preserves_alpha_at_split(self):
        parts, source = make_parts('bottomwear', 'legwear')
        size = (109, 91)
        restored_garment = restore_parts_to_source(parts, source.shape[:2], size)[0]
        restored_rgba = restore_source_canvas(parts[0]['img'], size)
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / 'character.psd')
            dump_parts_psd({}, source.shape[:2], target, part_dict_list=parts, source=source,
                           deleted=np.zeros((*size, 4), dtype=np.uint8))
            psd = PSDImage.open(target)
            self.assertEqual([layer.name for layer in psd], ['bottomwear-back', 'legwear', 'bottomwear-front', 'Deleted'])
            self.assertFalse(psd[3].visible)
            back = layer_on_canvas(psd[0], psd.size)
            front = layer_on_canvas(psd[2], psd.size)
            merged = over(front, back)
            np.testing.assert_allclose(merged[..., 3], restored_rgba[..., 3], atol=1, rtol=0)
            left, top, right, bottom = restored_garment['xyxy']
            merged = merged[top:bottom, left:right]
            visible = merged[..., 3] > 0
            np.testing.assert_allclose(merged[visible], restored_garment['img'][visible], atol=1, rtol=0)
