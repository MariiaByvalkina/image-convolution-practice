import numpy as np
from PIL import Image


class ImageConvolution:

    def __init__(self, kernel: np.ndarray, mode: str = "edge"):
        self.kernel: np.ndarray = np.array(kernel, dtype=np.float64)
        self.mode: str = mode

    def set_mode(self, mode: str) -> "ImageConvolution":
        self.mode = mode
        return self

    def _padding_edge(self, padded: np.ndarray, image: np.ndarray, padding_height: int, padding_width: int) -> np.ndarray:
        height, width = image.shape
        padded[:padding_height, padding_width:padding_width + width] = image[0, :]
        padded[padding_height + height:, padding_width: padding_width + width] = image[-1, :]
        for i in range(padding_width):
            padded[:, i] = padded[:, padding_width]
            padded[:, -i - 1] = padded[:, -padding_width - 1]
        return padded

    def _padding_reflect(self, padded: np.ndarray, image: np.ndarray, padding_height: int, padding_width: int) -> np.ndarray:
        height, width = image.shape
        padded[:padding_height, padding_width:padding_width + width] = image[1:padding_height + 1, :][::-1]
        padded[padding_height + height:, padding_width:padding_width + width] = image[-padding_height - 1:-1, :][::-1]
        for i in range(padding_width):
            padded[:, padding_width - 1 - i] = padded[:, padding_width + 1 + i]
            padded[:, padding_width + width + i] = padded[:, padding_width + width - 2 - i]
        return padded

    def _padding(self, image: np.ndarray) -> np.ndarray:
        kernel_height, kernel_width = self.kernel.shape
        padding_height, padding_width = kernel_height // 2, kernel_width // 2
        height, width = image.shape
        padded = np.zeros((height + 2 * padding_height, width + 2 * padding_width))
        padded[padding_height:padding_height + height, padding_width:padding_width + width] = image
        if self.mode == "edge":
            padded = self._padding_edge(padded, image, padding_height, padding_width)
        elif self.mode == "reflect":
            padded = self._padding_reflect(padded, image, padding_height, padding_width)
        else:
            raise ValueError(f"Неизвестный режим паддинга: '{self.mode}'.")
        return padded

    def convolve(self, image: Image.Image, normal: bool = True) -> np.ndarray:
        if image is None:
            raise ValueError("Изображение не загружено")
        if self.kernel is None:
            raise ValueError("Ядро не выбрано")

        image_array = np.array(image.convert('L'), dtype=np.float64) / 255.0

        padded_image = self._padding(image_array)

        kernel_height, kernel_width = self.kernel.shape
        height, width = image_array.shape

        output = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * self.kernel)

        if normal:
            output = np.clip(output, 0, 1)

        return (output * 255).astype(np.uint8)
