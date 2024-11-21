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
    
    def get_values_vectorized(self, i_vector: np.ndarray, j_vector: np.ndarray) -> np.ndarray:
        """
        Compute the value of the matrix at position (row, col) dynamically.
        """
        # Find which stacked Toeplitz matrix this row/col is in
        toeplitz_x = j_vector // self.single_toeplitz_width
        toeplitz_y = i_vector // self.single_toeplitz_height
        toeplitz_row = i_vector % self.single_toeplitz_height
        toeplitz_col = j_vector % self.single_toeplitz_width

        padded_kernel_row = (self.kernel.shape[0] - 1) - (toeplitz_y - toeplitz_x) # Not sure why this needed to be -2 instead of -1
        padded_kernel_col = toeplitz_row - toeplitz_col

        # If the row/col is outside the kernel, return 0
        if (padded_kernel_row < 0
            or padded_kernel_row >= self.kernel.shape[0]
            or padded_kernel_col < 0
            or padded_kernel_col >= self.kernel.shape[1]):
            return 0.0

        return self.kernel[padded_kernel_row, padded_kernel_col]

    def get_nonzero_rows(self, j: int) -> np.ndarray:
        """
        Get the indices of nonzero elements in the j-th column of the matrix.
        """
        # Toeplitz block column index
        tb_j = j // self.single_toeplitz_width

        # Column of inner Toeplitz
        t_j = j % self.single_toeplitz_width

        # Compute the range for each inner Toeplitz block
        inner_rows = np.arange(t_j, t_j + self.kernel.shape[1])

        # Generate all indices for the inner Toeplitz rows
        # The toeplitzes are all empty up until the first diagonal
        start_row = tb_j * self.single_toeplitz_height
        end_row = start_row + self.single_toeplitz_height * self.kernel.shape[0]
        nonzero_rows = (
            np.arange(start_row, end_row, self.single_toeplitz_height)[:, None] + inner_rows
        ).flatten()

        return nonzero_rows
    
    def get_nonzero_rows_vectorized(self, j_vector: np.ndarray) -> np.ndarray:
        """
        Get the indices of nonzero elements in the j-th column of the matrix.
        """
        assert j_vector.ndim == 1
        if j_vector.size == 0:
            return np.array([], dtype=int)
        
        # TODO: Calculate how much memory is being used by this function
        # TODO: See if this stuff can be done more in-place by using np.add.at/numpy.ufunc.at
        # and freeing up memory by overwriting existing arrays or something
        
        # Toeplitz block column indices of each column (1-dimensional)
        tb_j_vector = j_vector // self.single_toeplitz_width

        # Column indices relative to inner Toeplitzes (1-dimensional)
        t_j_vector = j_vector % self.single_toeplitz_width

        # Compute the range of rows for each inner Toeplitz block (2-dimensional)
        t_i_matrix = t_j_vector[:, None] + np.arange(self.kernel.shape[1])

        # Generate all indices for the inner Toeplitz rows
        i_range = self.single_toeplitz_height * self.kernel.shape[0]
        
        # Repeat the toeplitz-relative row indices for each vertically stacked nonzero toeplitz
        i_matrix = np.tile(t_i_matrix, self.kernel.shape[0])
        i_offsets_slice = np.arange(0, i_range, self.single_toeplitz_height)
        i_offsets_slice = np.repeat(i_offsets_slice, self.kernel.shape[1])
        # Broadcast toeplitz-height row offsets to the i_matrix
        i_matrix = i_matrix + i_offsets_slice

        # Broadcast diagonal row offsets across the i_matrix
        diagonal_row_offsets = tb_j_vector[:, None] * self.single_toeplitz_height
        i_matrix = i_matrix + diagonal_row_offsets

        return i_matrix

