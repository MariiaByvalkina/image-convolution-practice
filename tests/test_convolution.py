import pytest
import numpy as np
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from convolution import ImageConvolution
from kernels import get_kernel

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")
os.makedirs(GOLDEN_DIR, exist_ok=True)

kernels = ["sharpen", "blur", "sobel"]
modes = ["zero", "edge", "reflect"]

@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test.png"
    data = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
    Image.fromarray(data).save(path)
    return str(path)

@pytest.mark.parametrize("k_name", kernels)
@pytest.mark.parametrize("mode", modes)
def test_convolution_integrity(sample_image, k_name, mode):

    golden_path = os.path.join(GOLDEN_DIR, f"{k_name}_{mode}.png")

    kernel = get_kernel(k_name)
    tool = ImageConvolution(kernel, mode)
    tool.image_to_BW(sample_image).convolve()

    result = (tool.result() * 255).astype(np.uint8)

    if not os.path.exists(golden_path):
        Image.fromarray(result).save(golden_path)
        pytest.skip(f"New golden created: {golden_path}")

    expected = np.array(Image.open(golden_path))

    assert np.array_equal(result, expected), f"Расхождение в режиме {k_name}_{mode}"
