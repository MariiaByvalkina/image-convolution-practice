import numpy as np
from PIL import Image
from typing import Optional


class ImageConvolution:

    def __init__(self, kernel: np.ndarray, mode: str = "zero"):
        self.kernel: np.ndarray = np.array(kernel, dtype=np.float64)
        self.image: Optional[np.ndarray] = None
        self.mode: str = mode
        self.result_data: Optional[np.ndarray] = None

    def set_mode(self, mode: str) -> "ImageConvolution":
        self.mode = mode
        return self

    def image_to_BW(self, image: str) -> "ImageConvolution":
        img = Image.open(image).convert("L")
        self.image = np.array(img, dtype=np.float64) / 255.0
        return self

    def padding(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("Изображение не загружено")

        kernel_height, kernel_width = self.kernel.shape
        padding_height, padding_width = kernel_height // 2, kernel_width // 2
        height, width = self.image.shape

        padded = np.zeros((height + 2 * padding_height, width + 2 * padding_width))
        padded[padding_height:padding_height + height, padding_width:padding_width + width] = self.image

        if self.mode == "zero":
            pass

        if self.mode == "edge":
            padded[:padding_height, padding_width:padding_width + width] = self.image[0, :]
            padded[padding_height + height:, padding_width: padding_width + width] = self.image[-1, :]

            for i in range(padding_width):
                padded[:, i] = padded[:, padding_width]
                padded[:, -i - 1] = padded[:, -padding_width - 1]

        if self.mode == "reflect":
            padded[:padding_height, padding_width:padding_width + width] = self.image[1:padding_height + 1, :][::-1]
            padded[padding_height + height:, padding_width:padding_width + width] = self.image[-padding_height - 1:-1, :][::-1]

            for i in range(padding_width):
                padded[:, padding_width - 1 - i] = padded[:, padding_width + 1 + i]
                padded[:, padding_width + width + i] = padded[:, padding_width + width - 2 - i]

        return padded

    def convolve(self, normal: bool = True) -> "ImageConvolution":
        if self.image is None:
            raise ValueError("Изображение не загружено")
        if self.kernel is None:
            raise ValueError("Ядро не выбрано")

        padded_image = self.padding()
        kernel_height, kernel_width = self.kernel.shape
        height, width = self.image.shape

        output = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * self.kernel)

        if normal:
            output = np.clip(output, 0, 1)

        self.result_data = output
        return self

    def save_result(self, path: str) -> "ImageConvolution":
        if self.result_data is None:
            raise ValueError("Нет результата для сохранения")

        uint8_array = (self.result_data * 255).astype(np.uint8)
        result_image = Image.fromarray(uint8_array, mode="L")
        result_image.save(path)
        return self

    def result(self) -> Optional[np.ndarray]:
        return self.result_data
