import numpy as np
from PIL import Image, ImageFilter
import cv2
import pytest

from convolution import ImageConvolution
from kernels import get_kernel

IMAGE_SIZES = [512, 1024]
KERNELS = ["blur", "sharpen", "sobel"]

@pytest.fixture(params=IMAGE_SIZES)
def gray_pil_image(request):
    size = request.param
    image = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    return Image.fromarray(image, mode="L")


@pytest.fixture(params=IMAGE_SIZES)
def rgb_pil_image(request):
    size = request.param
    image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


@pytest.fixture(params=IMAGE_SIZES)
def gray_cv_image(request):
    size = request.param
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)


@pytest.fixture(params=IMAGE_SIZES)
def rgb_cv_image(request):
    size = request.param
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

#-----Own grayscale-----

def test_own_grayscale_sharpen(benchmark, gray_pil_image):
    tool = ImageConvolution(get_kernel("sharpen"), "edge")
    benchmark(tool.convolve, gray_pil_image)


def test_own_grayscale_blur(benchmark, gray_pil_image):
    tool = ImageConvolution(get_kernel("blur"), "edge")
    benchmark(tool.convolve, gray_pil_image)


def test_own_grayscale_sobel(benchmark, gray_pil_image):
    tool = ImageConvolution(get_kernel("sobel"), "edge")
    benchmark(tool.convolve, gray_pil_image)


#----Own RGB----

def test_own_rgb_sharpen(benchmark, rgb_pil_image):
    tool = ImageConvolution(get_kernel("sharpen"), "edge")
    benchmark(tool.convolve, rgb_pil_image)


def test_own_rgb_blur(benchmark, rgb_pil_image):
    tool = ImageConvolution(get_kernel("blur"), "edge")
    benchmark(tool.convolve, rgb_pil_image)


def test_own_rgb_sobel(benchmark, rgb_pil_image):
    tool = ImageConvolution(get_kernel("sobel"), "edge")
    benchmark(tool.convolve, rgb_pil_image)


#----Pillow----

def test_pillow_grayscale_sharpen(benchmark, gray_pil_image):
    benchmark(gray_pil_image.filter, ImageFilter.SHARPEN)


def test_pillow_grayscale_blur(benchmark, gray_pil_image):
    benchmark(gray_pil_image.filter, ImageFilter.BLUR)


def test_pillow_rgb_sharpen(benchmark, rgb_pil_image):
    benchmark(rgb_pil_image.filter, ImageFilter.SHARPEN)


def test_pillow_rgb_blur(benchmark, rgb_pil_image):
    benchmark(rgb_pil_image.filter, ImageFilter.BLUR)


#----OpenCV----

def test_opencv_grayscale_sharpen(benchmark, gray_cv_image):
    kernel = np.array(get_kernel("sharpen"), dtype=np.float32)
    benchmark(cv2.filter2D, gray_cv_image, -1, kernel)


def test_opencv_grayscale_blur(benchmark, gray_cv_image):
    kernel = np.array(get_kernel("blur"), dtype=np.float32)
    benchmark(cv2.filter2D, gray_cv_image, -1, kernel)


def test_opencv_grayscale_sobel(benchmark, gray_cv_image):
    kernel = np.array(get_kernel("sobel"), dtype=np.float32)
    benchmark(cv2.filter2D, gray_cv_image, -1, kernel)


def test_opencv_rgb_sharpen(benchmark, rgb_cv_image):
    kernel = np.array(get_kernel("sharpen"), dtype=np.float32)
    benchmark(cv2.filter2D, rgb_cv_image, -1, kernel)


def test_opencv_rgb_blur(benchmark, rgb_cv_image):
    kernel = np.array(get_kernel("blur"), dtype=np.float32)
    benchmark(cv2.filter2D, rgb_cv_image, -1, kernel)


def test_opencv_rgb_sobel(benchmark, rgb_cv_image):
    kernel = np.array(get_kernel("sobel"), dtype=np.float32)
    benchmark(cv2.filter2D, rgb_cv_image, -1, kernel)
