from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import cv2

from convolution import ImageConvolution
from kernels import get_kernel

INPUT_IMAGE_PATH = Path(__file__).parent.parent / "data" / "input.jpg"


def load_grayscale() -> Image.Image:
    return Image.open(INPUT_IMAGE_PATH).convert("L")


def load_rgb() -> Image.Image:
    return Image.open(INPUT_IMAGE_PATH).convert("RGB")

#-----Own grayscale-----

def test_own_grayscale_sharpen(benchmark):
    image = load_grayscale()
    tool = ImageConvolution(get_kernel("sharpen"), "edge")
    benchmark(tool.convolve, image)


def test_own_grayscale_blur(benchmark):
    image = load_grayscale()
    tool = ImageConvolution(get_kernel("blur"), "edge")
    benchmark(tool.convolve, image)


def test_own_grayscale_sobel(benchmark):
    image = load_grayscale()
    tool = ImageConvolution(get_kernel("sobel"), "edge")
    benchmark(tool.convolve, image)


#----Own RGB----

def test_own_rgb_sharpen(benchmark):
    image = load_rgb()
    tool = ImageConvolution(get_kernel("sharpen"), "edge")
    benchmark(tool.convolve, image)


def test_own_rgb_blur(benchmark):
    image = load_rgb()
    tool = ImageConvolution(get_kernel("blur"), "edge")
    benchmark(tool.convolve, image)


def test_own_rgb_sobel(benchmark):
    image = load_rgb()
    tool = ImageConvolution(get_kernel("sobel"), "edge")
    benchmark(tool.convolve, image)


#----Pillow----

def test_pillow_grayscale_sharpen(benchmark):
    image = load_grayscale()
    benchmark(image.filter, ImageFilter.SHARPEN)


def test_pillow_grayscale_blur(benchmark):
    image = load_grayscale()
    benchmark(image.filter, ImageFilter.BLUR)


def test_pillow_rgb_sharpen(benchmark):
    image = load_rgb()
    benchmark(image.filter, ImageFilter.SHARPEN)


def test_pillow_rgb_blur(benchmark):
    image = load_rgb()
    benchmark(image.filter, ImageFilter.BLUR)


#----OpenCV----

def test_opencv_grayscale_sharpen(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    kernel = np.array(get_kernel("sharpen"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)


def test_opencv_grayscale_blur(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    kernel = np.array(get_kernel("blur"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)


def test_opencv_grayscale_sobel(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    kernel = np.array(get_kernel("sobel"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)


def test_opencv_rgb_sharpen(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_COLOR)
    kernel = np.array(get_kernel("sharpen"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)


def test_opencv_rgb_blur(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_COLOR)
    kernel = np.array(get_kernel("blur"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)


def test_opencv_rgb_sobel(benchmark):
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_COLOR)
    kernel = np.array(get_kernel("sobel"), dtype=np.float32)
    benchmark(cv2.filter2D, image, -1, kernel)
