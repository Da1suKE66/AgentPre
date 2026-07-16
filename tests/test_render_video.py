from __future__ import annotations

import unittest

import numpy as np

from src.render_video import VideoRenderError, sample_frame_indices, unpack_rgb


class RenderVideoTests(unittest.TestCase):
    def test_samples_60hz_trajectory_at_30fps_without_interpolation(self) -> None:
        indices, stride = sample_frame_indices(1336, 1.0 / 60.0, 30.0)
        self.assertEqual(stride, 2)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1334)
        self.assertEqual(len(indices), 668)
        np.testing.assert_array_equal(indices[:4], [0, 2, 4, 6])

    def test_rejects_non_integer_stride_fps(self) -> None:
        with self.assertRaises(VideoRenderError):
            sample_frame_indices(120, 1.0 / 60.0, 24.0)

    def test_unpacks_newton_rgba_uint32_to_rgb24(self) -> None:
        packed = np.asarray([[0xFF332211, 0xFFCCBBAA]], dtype=np.uint32)
        rgb = unpack_rgb(packed)
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(rgb.flags.c_contiguous)
        np.testing.assert_array_equal(rgb.tolist(), [[[0x11, 0x22, 0x33], [0xAA, 0xBB, 0xCC]]])

    def test_unpack_requires_2d_uint32(self) -> None:
        with self.assertRaises(VideoRenderError):
            unpack_rgb(np.zeros((2, 2), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
