import numpy as np
from typing import Tuple
from scipy.sparse import spmatrix

# TODO: This is all straight from ChatGPT. I need to go through it and verify/fix it.

class ToeplitzSparseMatrix(spmatrix):
    def __init__(self, input_size: Tuple[int, int], kernel_size: Tuple[int, int], kernel_values: np.ndarray):
        """
        Initialize the Toeplitz-like sparse matrix.

        Args:
            input_size (tuple): Dimensions of the input matrix (rows, cols).
            kernel_size (tuple): Dimensions of the kernel (rows, cols).
            kernel_values (np.ndarray): Values of the kernel.
        """
        self.input_size = input_size  # Size of the input matrix (e.g., 10,000 x 10,000)
        self.kernel_size = kernel_size  # Size of the kernel (e.g., 10,000 x 10,000)
        self.kernel_values = kernel_values  # Kernel values as a 2D array
        self.shape = (input_size[0] * input_size[1], input_size[0] * input_size[1])

    def _compute_value(self, row, col):
        """
        Compute the value of the matrix at position (row, col) dynamically.
        """
        # Find which stacked Toeplitz matrix this row/col is in
        toeplitz_matrix_width = self.input_size[1]
        toeplitz_matrix_height = self.input_size[0] + self.kernel_size[0] - 1

        toeplitz_x = col // toeplitz_matrix_width
        toeplitz_y = row // toeplitz_matrix_height
        padded_kernel_row = (self.kernel_size[0] - 1) - toeplitz_y + toeplitz_x
        padded_kernel_col = col % toeplitz_matrix_width - row % toeplitz_matrix_height

        # If the row/col is outside the kernel, return 0
        if padded_kernel_row < 0 or padded_kernel_row >= self.kernel_size[0] or padded_kernel_col < 0 or padded_kernel_col >= self.kernel_size[1]:
            return 0.0

        return self.kernel_values[padded_kernel_row, padded_kernel_col]

    def __getitem__(self, index):
        """
        Retrieve the value at the given index.
        Args:
            index (tuple): (row, col) for the matrix entry.
        Returns:
            float: The matrix value.
        """
        row, col = index
        if row < 0 or col < 0 or row >= self.shape[0] or col >= self.shape[1]:
            raise IndexError("Index out of bounds")
        return self._compute_value(row, col)

    def __matmul__(self, vector):
        """
        Matrix-vector multiplication (overriding the @ operator).
        Args:
            vector (np.ndarray): The input vector.
        Returns:
            np.ndarray: Result of the multiplication.
        """
        if vector.shape[0] != self.shape[1]:
            raise ValueError("Dimension mismatch for matrix-vector multiplication")
        
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            for j in range(self.kernel_size[0]):
                kernel_value = self.kernel_values[j % self.kernel_size[0], j % self.kernel_size[1]]
                result[i] += kernel_value * vector[(i - j) % self.shape[1]]
        
        return result

    def toarray(self):
        """
        Convert to a dense array (use only for testing or small cases!).
        """
        dense = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                dense[i, j] = self._compute_value(i, j)
        return dense

    def todense(self):
        """
        Alias for `toarray`.
        """
        return self.toarray()
