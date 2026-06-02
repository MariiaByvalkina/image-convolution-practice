from pathlib import Path

import pytest
import numpy as np
import os
from PIL import Image

from convolution import ImageConvolution
from kernels import get_kernel

INPUT_IMAGE_PATH = Path(__file__).parent.parent / "data"
GOLDEN_DIR = Path(__file__).parent / "goldens"

os.makedirs(GOLDEN_DIR, exist_ok=True)

kernels = ["sharpen", "blur", "sobel"]
modes = ["edge", "reflect"]

@pytest.fixture(params=list(INPUT_IMAGE_PATH.glob("*.jpg")))
def sample_image(request):
    return request.param.stem, Image.open(request.param)


@pytest.mark.parametrize("kernel_name", kernels)
@pytest.mark.parametrize("mode", modes)
def test_convolution_integrity(sample_image, kernel_name, mode):
    image_name, image = sample_image
    golden_path = GOLDEN_DIR / f"{image_name}_{kernel_name}_{mode}.png"

    kernel = get_kernel(kernel_name)
    tool = ImageConvolution(kernel, mode)
    result = tool.convolve(image)

    if not golden_path.exists():
        Image.fromarray(result.astype(np.uint8)).save(golden_path)
        pytest.skip(f"Golden created: {golden_path}")

    expected = np.array(Image.open(golden_path))
    assert np.array_equal(result, expected), f"Расхождение в {image_name}_{kernel_name}_{mode}"

