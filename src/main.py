import argparse
import numpy as np
from PIL import Image

from src.kernels import get_kernel
from src.convolution import ImageConvolution


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert('L')
    return np.array(image, dtype=np.float64) / 255.0


def save_result(image: np.ndarray, path: str) -> None:
    img = Image.fromarray((image * 255).astype(np.uint8), mode="L")
    img.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--mode", choices=["edge", "reflect"], default="edge")
    parser.add_argument("--kernel", choices=["sharpen", "blur", "sobel"], default="sharpen")
    parser.add_argument("--out", default="result.png")

    args = parser.parse_args()

    new_kernel = get_kernel(args.kernel)

    tool = ImageConvolution(new_kernel, mode=args.mode)
    result = tool.convolve(load_image(args.input))
    save_result(result, args.out)
