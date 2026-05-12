from __future__ import annotations

import numpy as np
import pytest

from mmsec_eval.model_adapters.image_utils import (
    bchw_to_pil_images,
    image_to_pil_rgb,
    image_to_rgb01,
    resize_rgb01,
)


def test_image_to_rgb01_accepts_grayscale_and_uint8_scale():
    image = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    out = image_to_rgb01(image)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert out[0, 1, 0] == pytest.approx(128.0 / 255.0)


def test_image_to_rgb01_accepts_single_channel_and_rgba():
    single = np.ones((3, 2, 1), dtype=np.float32) * 0.25
    rgba = np.dstack(
        [
            np.ones((3, 2), dtype=np.float32) * 0.1,
            np.ones((3, 2), dtype=np.float32) * 0.2,
            np.ones((3, 2), dtype=np.float32) * 0.3,
            np.ones((3, 2), dtype=np.float32) * 0.9,
        ]
    )

    assert image_to_rgb01(single).shape == (3, 2, 3)
    rgb = image_to_rgb01(rgba)

    assert rgb.shape == (3, 2, 3)
    assert rgb[0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_image_to_rgb01_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="unsupported image shape"):
        image_to_rgb01(np.zeros((1, 2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="image array is empty"):
        image_to_rgb01(np.zeros((0, 0), dtype=np.float32))


def test_resize_rgb01_and_pil_conversion_are_stable():
    image = np.random.default_rng(7).random((8, 6, 3), dtype=np.float32)

    resized = resize_rgb01(image, size_hw=(4, 5))
    pil = image_to_pil_rgb(resized)

    assert resized.shape == (4, 5, 3)
    assert resized.dtype == np.float32
    assert pil.mode == "RGB"
    assert pil.size == (5, 4)


def test_bchw_to_pil_images_preserves_batch_order():
    batch = np.zeros((2, 3, 4, 5), dtype=np.float32)
    batch[1, 0, :, :] = 1.0

    images = bchw_to_pil_images(batch)

    assert len(images) == 2
    assert images[0].size == (5, 4)
    assert images[1].getpixel((0, 0))[0] == 255
