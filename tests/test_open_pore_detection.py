import unittest

import numpy as np

from advanced_tools.three_d_nmr.open_pore_workflow import detect_open_pores


class OpenPoreDetectionTest(unittest.TestCase):
    def test_only_boundary_connected_pore_is_open(self):
        volume = np.full((5, 5, 5), 255, dtype=np.uint8)
        volume[0, 1, 1] = 0
        volume[1, 1, 1] = 0
        volume[3, 3, 3] = 0

        result = detect_open_pores(volume, pore_value=0, solid_value=255, connectivity=6)

        self.assertTrue(result.open_mask[0, 1, 1])
        self.assertTrue(result.open_mask[1, 1, 1])
        self.assertFalse(result.open_mask[3, 3, 3])
        self.assertEqual(result.total_pore_voxels, 3)
        self.assertEqual(result.open_pore_voxels, 2)
        self.assertEqual(result.closed_pore_voxels, 1)
        self.assertEqual(result.pore_component_count, 2)


if __name__ == "__main__":
    unittest.main()
