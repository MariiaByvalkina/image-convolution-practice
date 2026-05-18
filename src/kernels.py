import numpy as np


def get_kernel(kernel: str) -> np.ndarray:
    kernels = {
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "blur": np.array(
            [
                [
                    1,
                    1,
                    1,
                ],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        / 9,
        "sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    }
    return kernels.get(kernel, kernels["sharpen"])
