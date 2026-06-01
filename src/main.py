import argparse
from PIL import Image

from kernels import get_kernel
from convolution import ImageConvolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--mode", choices=["edge", "reflect"], default="edge")
    parser.add_argument("--kernel", choices=["sharpen", "blur", "sobel"], default="sharpen")
    parser.add_argument("--out", default="result.png")
    args = parser.parse_args()

    image = Image.open(args.input)
    new_kernel = get_kernel(args.kernel)
    tool = ImageConvolution(new_kernel, mode=args.mode)
    result = tool.convolve(image)
    Image.fromarray(result, mode='L').save(args.out)
