import numpy as np
from typing import Tuple
from scipy.sparse import spmatrix

# TODO: This is all straight from ChatGPT. I need to go through it and verify/fix it.

class DoubleToeplitzMatrix(spmatrix):
    def __init__(self, input_size: Tuple[int, int, int], kernel_size: Tuple[int, int], kernel_values: np.ndarray):
        """
        Initialize the Toeplitz-like sparse matrix.

        Args:
            input_size (tuple): Dimensions of the input matrix (n, rows, cols).
            kernel_size (tuple): Dimensions of the kernel (rows, cols).
            kernel_values (np.ndarray): Values of the kernel.
        """
        self.input_size = input_size  # (batch_size, rows, cols)
        self.kernel_size = kernel_size  # (rows, cols)
        self.kernel_values = kernel_values  # Kernel values as a 2D array

        batch_size, input_rows, input_cols = input_size
        self.padded_kernel_height = input_rows + kernel_size[0] - 1
        self.padded_kernel_width = input_cols + kernel_size[1] - 1

        self.single_toeplitz_height = self.padded_kernel_width
        self.single_toeplitz_width = input_size[2]
        self._shape = (self.single_toeplitz_height * self.padded_kernel_height, self.single_toeplitz_width * input_size[1])

    @property
    def shape(self):
        """
        Return the shape of the matrix.
        """
        return self._shape

    def reshape(self, shape):
        """
        This method is required by the `spmatrix` class but is not implemented for this matrix type.
        """
        raise NotImplementedError("reshape is not implemented for DoubleToeplitzMatrix")

    def _compute_value(self, row, col):
        """
        Compute the value of the matrix at position (row, col) dynamically.
        """
        # Find which stacked Toeplitz matrix this row/col is in
        toeplitz_x = col // self.single_toeplitz_width
        toeplitz_y = row // self.single_toeplitz_height
        toeplitz_row = row % self.single_toeplitz_height
        toeplitz_col = col % self.single_toeplitz_width
        padded_kernel_row = self.padded_kernel_height - 2 - (toeplitz_y - toeplitz_x) # Not sure why this needed to be -2 instead of -1
        padded_kernel_col = toeplitz_row - toeplitz_col

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

    def __matmul__(self, B: np.ndarray):
        """
        Matrix multiplication (overriding the @ operator).
        Args:
            B (np.ndarray): The input matrix.
        Returns:
            np.ndarray: Result of the multiplication.
        """
        if B.shape[0] != self.shape[1]:
            raise ValueError("Dimension mismatch for matrix multiplication: {} vs {}".format(B.shape[0], self.shape[1]))

        result = np.zeros((self.shape[0], B.shape[1]))
        for X_row in range(self.shape[0]):
            for v_col in range(B.shape[1]):
                for v_row in range(B.shape[0]):
                    result[X_row, v_col] += self._compute_value(X_row, v_row) * B[v_row, v_col]
        
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

    def __str__(self):
        """
        Return a string representation of the DoubleToeplitzMatrix.
        Shows the top-left corner of the matrix for large dimensions.
        """
        # Generate the matrix representation
        matrix_str = []
        for row in range(self.shape[0]):
            row_str = []
            for col in range(self.shape[1]):
                row_str.append(f"{self._compute_value(row, col):.2f}\t")
            matrix_str.append(" ".join(row_str))

        # Combine rows into a single string
        return "\n".join(matrix_str)

