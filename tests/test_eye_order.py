from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))

from utils.inference_utils import dump_parts_psd, psd2partdicts


class EyeOrderTests(unittest.TestCase):
    def test_eyes_export_above_face_in_anatomical_order(self):
        for split_tags in ((), ('eyewhite', 'irides', 'eyelash'), ('irides',)):
            with self.subTest(split_tags=split_tags), tempfile.TemporaryDirectory() as tmp:
                size = (16, 32)
                parts = {}

                def add(tag, rgb, depth, boxes):
                    img = np.zeros((*size, 4), dtype=np.uint8)
                    for x1, y1, x2, y2 in boxes:
                        img[y1:y2, x1:x2] = (*rgb, 255)
                    parts[tag] = {'tag': tag, 'img': img, 'xyxy': [0, 0, size[1], size[0]],
                                  'depth_median': depth, 'depth': np.full(size, round(depth * 255), dtype=np.uint8)}

                add('face', (235, 148, 125), 0.4, [(0, 0, 32, 16)])
                add('front hair', (156, 73, 48), 0.1, [(3, 4, 4, 10)])
                eyes = [('eyewhite', (245, 245, 245), 0.2, [(3, 4, 12, 10), (19, 4, 28, 10)]),
                        ('irides', (60, 108, 148), 0.6, [(6, 4, 9, 9), (22, 4, 25, 9)]),
                        ('eyelash', (21, 24, 31), 0.8, [(3, 4, 12, 5), (19, 4, 28, 5)])]
                expected = ['face']
                for tag, rgb, depth, boxes in eyes:
                    if tag in split_tags:
                        for box, suffix in zip(boxes, ('-l', '-r')):
                            add(tag + suffix, rgb, depth, [box])
                        expected.extend([tag, tag])
                    else:
                        add(tag, rgb, depth, boxes)
                        expected.append(tag)
                expected.append('front hair')

                path = str(Path(tmp) / 'eyes.psd')
                dump_parts_psd(parts, size, path)
                psd = PSDImage.open(path)
                depth = PSDImage.open(Path(tmp) / 'eyes_depth.psd')
                names = [layer.name for layer in psd]
                kinds = [name.split('-', 1)[0] for name in names]
                self.assertEqual(kinds, expected)
                self.assertEqual([layer.name for layer in depth], names)
                art = np.asarray(psd.composite(force=True))
                for offset in (0, 16):
                    np.testing.assert_allclose(art[6, 7 + offset, :3], [60, 108, 148], atol=1, rtol=0)
                    np.testing.assert_allclose(art[4, 7 + offset, :3], [21, 24, 31], atol=1, rtol=0)
                    np.testing.assert_allclose(art[6, 10 + offset, :3], [245, 245, 245], atol=1, rtol=0)
                np.testing.assert_allclose(art[6, 3, :3], [156, 73, 48], atol=1, rtol=0)

                reloaded = psd2partdicts(path)
                reexport = str(Path(tmp) / 'reexport.psd')
                dump_parts_psd(reloaded['parts'], reloaded['frame_size'], reexport)
                self.assertEqual([layer.name for layer in PSDImage.open(reexport)], names)
