import numpy as np
from PIL import Image
from typing import Optional


class ImageConvolution:

    def __init__(self, kernel: np.ndarray, mode: str = "zero"):
        self.kernel: np.ndarray = np.array(kernel)
        self.image: Optional[np.ndarray] = None
        self.mode: str = mode
        self.result: Optional[np.ndarray] = None

    def set_mode(self, mode: str) -> "ImageConvolution":
        self.mode = mode
        return self

    def image_to_BW(self, image: str) -> "ImageConvolution":
        img = Image.open(image).convert("L")
        self.image = np.array(img) / 255
        return self

    def padding(self) -> np.ndarray:
        kernel_height, kernel_width = self.kernel.shape
        padding_height, padding_width = kernel_height // 2, kernel_width // 2

        height, width = self.image.shape

        padded = np.zeros((height + 2 * padding_height, width + 2 * padding_width))

        padded[
            padding_height : padding_height + height,
            padding_width : padding_width + width,
        ] = self.image

        if self.mode == "zero":
            pass

        if self.mode == "edge":
            padded[:padding_height, padding_width : padding_width + width] = self.image[
                0, :
            ]
            padded[: padding_height + height, padding_width : padding_width + width] = (
                self.image[-1, :]
            )

            for i in range(padding_width):
                padded[:, i] = padded[:, padding_width]
                padded[:, -i - 1] = padded[:, -padding_width - 1]

        if self.mode == "reflect":
            padded[:padding_height, padding_width : padding_width + width] = self.image[
                1 : padding_height + 1, :
            ][::-1]
            padded[padding_height + height :, padding_width : padding_width + width] = (
                self.image[-padding_height - 1 : -1, :][::-1]
            )

            for i in range(padding_width):
                padded[:, padding_width - 1 - i] = padded[:, padding_width + i + 1]
                padded[:, padding_width - width + i] = padded[
                    :, padding_width + width - 2 - i
                ]

        return padded

    def convolve(self, normal: bool = True) -> "ImageConvolution":
        padded_image = self.padding()
        kernel_height, kernel_width = self.kernel.shape
        height, width = self.image.shape

        output = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                region = padded_image[i : i + kernel_height, j : j + kernel_width]
                output[i, j] = np.sum(region * self.kernel)

        if normal:
            output = np.clip(output, 0, 1)

        self.result = output
        return self

    def save_result(self, path: str) -> "ImageConvolution":
        result_image = Image.fromarray((self.result * 255).astype(np.uint8))
        result_image.save(path)
        return self

    def result(self):
        return self.result
