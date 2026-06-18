import unittest

import numpy as np

from advanced_tools.three_d_nmr.true_3d_voxel_nmr import (
    VoxelNmrParams,
    build_3d_decay_system,
    build_3d_decay_system_matrix_free,
    matrix_free_linear_operator,
)


class True3DVoxelNmrTest(unittest.TestCase):
    def test_single_open_voxel_has_six_solid_boundary_faces(self):
        open_mask = np.zeros((3, 3, 3), dtype=bool)
        open_mask[1, 1, 1] = True
        params = VoxelNmrParams(voxel_size_um=2.0)

        system = build_3d_decay_system(open_mask, params)

        self.assertEqual(system.open_voxels, 1)
        self.assertEqual(system.internal_neighbor_pairs, 0)
        self.assertEqual(system.solid_boundary_faces, 6)
        self.assertEqual(system.external_boundary_faces, 0)
        self.assertAlmostEqual(system.pore_volume_um3, 8.0)

    def test_adjacent_open_voxels_share_internal_diffusion_face(self):
        open_mask = np.zeros((3, 3, 4), dtype=bool)
        open_mask[1, 1, 1] = True
        open_mask[1, 1, 2] = True
        params = VoxelNmrParams(voxel_size_um=1.0)

        system = build_3d_decay_system(open_mask, params)

        self.assertEqual(system.open_voxels, 2)
        self.assertEqual(system.internal_neighbor_pairs, 1)
        self.assertEqual(system.solid_boundary_faces, 10)

    def test_matrix_free_operator_matches_csr_operator(self):
        open_mask = np.zeros((4, 4, 4), dtype=bool)
        open_mask[1, 1, 1] = True
        open_mask[1, 1, 2] = True
        open_mask[1, 2, 2] = True
        params = VoxelNmrParams(voxel_size_um=1.7, dt_ms=5.0)

        csr_system = build_3d_decay_system(open_mask, params)
        matrix_free_system = build_3d_decay_system_matrix_free(open_mask, params)
        vector = np.array([1.0, 0.7, 0.3])

        np.testing.assert_allclose(
            matrix_free_linear_operator(matrix_free_system).matvec(vector),
            csr_system.lhs @ vector,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
