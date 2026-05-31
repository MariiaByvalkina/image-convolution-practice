import pytest
import numpy as np
import os
from PIL import Image

from convolution import ImageConvolution
from kernels import get_kernel

INPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "../data/input.jpg")
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")

os.makedirs(GOLDEN_DIR, exist_ok=True)

kernels = ["sharpen", "blur", "sobel"]
modes = ["edge", "reflect"]


@pytest.fixture
def sample_image():
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(
            f"Положите картинку по пути '{INPUT_IMAGE_PATH}'"
        )
    img = Image.open(INPUT_IMAGE_PATH).convert('L')
    return np.array(img, dtype=np.float64) / 255.0


@pytest.mark.parametrize("kernel_name", kernels)
@pytest.mark.parametrize("mode", modes)

def test_convolution_integrity(sample_image, kernel_name, mode):
    golden_path = os.path.join(GOLDEN_DIR, f"{kernel_name}_{mode}.png")

    kernel = get_kernel(kernel_name)
    tool = ImageConvolution(get_kernel(kernel_name), mode)
    tool.convolve(sample_image)
    result_uint8 = tool.get_result()

    if not os.path.exists(golden_path):
        Image.fromarray(result_uint8, mode="L").save(golden_path)
        pytest.skip(f"New golden created: {golden_path}")

    expected = np.array(Image.open(golden_path))
    assert np.array_equal(result_uint8, expected), f"Расхождение в режиме {kernel_name}_{mode}"

