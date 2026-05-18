import argparse
from kernels import get_kernel
from processor import ImageConvolution

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--mode", choices=['zero', 'edge', 'reflect'], default='zero')
    parser.add_argument("--kernel", choices=['sharpen', 'blur', 'sobel'], default='sharpen')
    parser.add_argument("--out", default="result.png")

    args = parser.parse_args()

    new_kernel = get_kernel(args.kernel)

    tool = ImageConvolution(kernel=new_kernel, mode=args.mode)
    tool.image_to_BW(args.input).convolve().save_result(args.out)
