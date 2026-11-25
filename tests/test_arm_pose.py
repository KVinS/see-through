from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))

from utils.arm_pose import arm_regions, complete_hidden_bone
from utils.clothing import _source_matte, split_clothing_layers
from utils.inference_utils import dump_parts_psd


def part(tag, img, depth):
    h, w = img.shape[:2]
    return {'tag': tag, 'img': img.copy(), 'xyxy': [0, 0, w, h],
            'depth_median': depth / 255, 'depth': np.full((h, w), depth, dtype=np.uint8)}


class ArmPoseTests(unittest.TestCase):
    def test_hidden_upper_arm_ignores_a_thin_extension_in_the_cover(self):
        size = (108, 100)
        image = np.full((*size, 4), [238, 169, 136, 255], dtype=np.uint8)
        visible = np.zeros(size, dtype=np.uint8)
        visible[18:88, 35:51] = 255
        cover = np.zeros_like(visible)
        cover[12:94, 51:57] = 255
        cover[32:34, 57:65] = 255
        cover[42, 52] = 250
        alpha, (paint, _) = complete_hidden_bone(image, visible, cover, np.array([51, 20]), np.array([51, 82]))
        self.assertGreater(alpha[40, 54], 0)
        self.assertEqual(alpha[42, 52], 255)
        self.assertEqual(alpha[32, 60], 0)
        self.assertTrue(np.all(cover[paint] >= 250))
        np.testing.assert_array_equal(alpha[visible > 0], visible[visible > 0])

    def test_cloth_boundary_uses_mixed_pixels_from_the_original(self):
        size = (64, 64)
        cloth = np.full((*size, 4), [32, 73, 180, 255], dtype=np.uint8)
        leg = np.full_like(cloth, [222, 155, 98, 255])
        source = cloth.copy()
        source[33:] = leg[33:]
        for row, fraction in ((31, 0.75), (32, 0.25)):
            source[row] = np.rint(cloth[row] * fraction + leg[row] * (1 - fraction)).astype(np.uint8)
        garment, limb = part('bottomwear', cloth, 170), part('legwear', leg, 52)
        result = split_clothing_layers([garment, limb], source)
        front = next(layer for layer in result if layer['tag'] == 'bottomwear-front')
        self.assertAlmostEqual(int(front['img'][31, 24, 3]), 191, delta=2)
        self.assertAlmostEqual(int(front['img'][32, 24, 3]), 64, delta=2)

        source[32, 24, :3] = (0, 255, 0)
        cut = np.indices(size)[0] < 32
        _, confidence = _source_matte(garment, [limb], source, cut, np.ones(size, dtype=bool))
        self.assertEqual(confidence[32, 24], 0)

    def test_sleeve_fits_between_upper_arm_and_forearm_without_cutting_the_sleeve(self):
        arm = np.zeros((112, 80, 4), dtype=np.uint8)
        sleeve = np.zeros_like(arm)
        arm[10:105, 32:48] = (240, 162, 130, 255)
        sleeve[27:80, 23:56] = (55, 111, 171, 255)
        source = arm.copy()
        source[sleeve[..., 3] > 0] = sleeve[sleeve[..., 3] > 0]
        source[53:80, 32:48] = arm[53:80, 32:48]
        parts = [part('handwear', arm, 90), part('topwear', sleeve, 165)]
        joints = {'r': np.array([[40, 19], [40, 52], [40, 82]])}
        with tempfile.TemporaryDirectory() as tmp, patch('utils.clothing.estimate_arm_joints', return_value=joints):
            path = str(Path(tmp) / 'arm.psd')
            dump_parts_psd({}, source.shape[:2], path, part_dict_list=parts, source=source)
            psd = PSDImage.open(path)
            self.assertEqual([layer.name for layer in psd],
                             ['handwear-r-upperarm', 'topwear', 'handwear-r-forearm', 'handwear-r-hand'])
            self.assertGreater(np.asarray(psd[0].topil())[55 - psd[0].top, 40 - psd[0].left, 3], 0)
            self.assertGreater(np.asarray(psd[2].topil())[85 - psd[2].top, 40 - psd[2].left, 3], 0)
            composite = np.asarray(psd.composite(force=True))
            for point in ((43, 40), (67, 40), (95, 40)):
                np.testing.assert_allclose(composite[point], source[point], atol=1, rtol=0)

    def test_folded_forearm_uses_the_drawn_crease_and_requires_visible_support(self):
        arm = np.zeros((120, 80, 4), dtype=np.uint8)
        arm[20:114, 19:59] = (238, 169, 136, 255)
        joints = {'l': np.array([[27, 29], [40, 105], [42, 48]])}
        self.assertEqual(list(arm_regions(part('handwear', arm, 100), joints)), [])
        arm[24:99, 34, :3] = (55, 37, 30)
        [(side, component, distal, hand, _)] = list(arm_regions(part('handwear', arm, 100), joints))
        self.assertEqual(side, 'l')
        self.assertTrue(component[72, 29])
        self.assertFalse(distal[72, 29])
        self.assertTrue(distal[72, 37])
        self.assertFalse(hand[72, 37])
        self.assertTrue(hand[29, 49])

    def test_thin_headwear_is_ordered_using_the_original_occlusion(self):
        headwear = np.zeros((64, 64, 4), dtype=np.uint8)
        hair = np.zeros_like(headwear)
        face = np.zeros_like(headwear)
        headwear[25, 8:56] = (242, 199, 71, 255)
        hair[10:46, 9:29] = (99, 29, 34, 255)
        face[15:51, 6:59] = (242, 164, 146, 255)
        for tucked in (False, True):
            with self.subTest(tucked=tucked), tempfile.TemporaryDirectory() as tmp:
                source = face.copy()
                source[hair[..., 3] > 0] = hair[hair[..., 3] > 0]
                visible = (headwear[..., 3] > 0) & ((hair[..., 3] == 0) | (not tucked))
                source[visible] = headwear[visible]
                parts = [part('face', face, 150), part('front hair', hair, 80), part('headwear', headwear, 190)]
                path = str(Path(tmp) / 'head.psd')
                dump_parts_psd({}, source.shape[:2], path, part_dict_list=parts, source=source)
                psd = PSDImage.open(path)
                expected = ['face', 'headwear', 'front hair'] if tucked else ['face', 'front hair', 'headwear']
                self.assertEqual([layer.name for layer in psd], expected)
