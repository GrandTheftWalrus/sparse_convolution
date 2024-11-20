import numpy as np
from typing import Tuple, List

class DoubleToeplitzHelper():
    def __init__(self, input_matrix_shape: Tuple[int, int], kernel: np.ndarray) -> None:
        self.input_matrix_shape: Tuple[int, int] = input_matrix_shape
        # self.kernel: np.ndarray = np.flip(kernel) # TODO: Make sure that flipping the kernel is correct
        self.kernel: np.ndarray = kernel # TODO: Make sure that flipping the kernel is correct
        self.padded_kernel_height: int = input_matrix_shape[0] + kernel.shape[0] - 1
        self.padded_kernel_width: int = input_matrix_shape[1] + kernel.shape[1] - 1

        self.single_toeplitz_height: int = self.padded_kernel_width
        self.single_toeplitz_width: int = input_matrix_shape[1]
        self.shape: Tuple[int, int] = (self.single_toeplitz_height * self.padded_kernel_height, self.single_toeplitz_width * input_matrix_shape[0])

    def get_value(self, row: int, col: int) -> float:
        """
        Compute the value of the matrix at position (row, col) dynamically.
        """
        # Find which stacked Toeplitz matrix this row/col is in
        toeplitz_x = col // self.single_toeplitz_width
        toeplitz_y = row // self.single_toeplitz_height
        toeplitz_row = row % self.single_toeplitz_height
        toeplitz_col = col % self.single_toeplitz_width

        padded_kernel_row = (self.kernel.shape[0] - 1) - (toeplitz_y - toeplitz_x) # Not sure why this needed to be -2 instead of -1
        padded_kernel_col = toeplitz_row - toeplitz_col

        # If the row/col is outside the kernel, return 0
        if (padded_kernel_row < 0
            or padded_kernel_row >= self.kernel.shape[0]
            or padded_kernel_col < 0
            or padded_kernel_col >= self.kernel.shape[1]):
            return 0.0

        return self.kernel[padded_kernel_row, padded_kernel_col]

    def get_nonzero_rows(self, j: int) -> List[int]:
        """
        Get the indices of nonzero elements in the j-th column of the matrix.
        """
        # Toeplitz block column index
        tb_j = j // self.single_toeplitz_width

        # Column of inner Toeplitz
        t_j = j % self.single_toeplitz_width

        # The toeplitzes are all empty up until the first diagonal
        start_row = tb_j * self.single_toeplitz_height
        indices: List[Tuple[int, int]] = []
        # For each nonzero inner toeplitz
        for i in range(start_row, self.shape[0], self.single_toeplitz_height):
            # (We are now at the top of a nonzero inner toeplitz)
            inner_start_row = t_j
            for i_inner in range(inner_start_row, inner_start_row + self.kernel.shape[1]):
                # Final indices
                indices.append(i + i_inner)

        return indices

