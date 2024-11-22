import numpy as np
from typing import Tuple
import scipy.sparse

class MinimalToeplitzConvolver():
    def __init__(self, input_matrix_shape: Tuple[int, int], kernel: np.ndarray) -> None:
        self.input_matrix_shape: Tuple[int, int] = input_matrix_shape
        self.kernel: np.ndarray = kernel
        self.padded_kernel_height: int = input_matrix_shape[0] + kernel.shape[0] - 1
        self.padded_kernel_width: int = input_matrix_shape[1] + kernel.shape[1] - 1

        self.single_toeplitz_height: int = self.padded_kernel_width
        self.single_toeplitz_width: int = input_matrix_shape[1]
        self.shape: Tuple[int, int] = (self.single_toeplitz_height * self.padded_kernel_height, self.single_toeplitz_width * input_matrix_shape[0])

    @staticmethod
    def convolve(input_matrices: np.ndarray, kernel: np.ndarray, mode) -> np.ndarray:
        """
        Perform a convolution operation between the input matrix and the kernel.
        """
        input_matrix_shape = input_matrices.shape[1:]
        k_shape = kernel.shape
        batch_size = input_matrices.shape[0]
        assert kernel.shape[0] > 0 and kernel.shape[1] > 0, "Kernel must have width and height greater than zero"
        assert input_matrix_shape[0] > 0 and input_matrix_shape[1] > 0, "Input matrix must have width and height greater than zero"

        # Make B (the flattened and horizontally stacked input matrices)
        B = np.flip(input_matrices, axis=1).reshape(batch_size, -1).T
        # TODO: get batching to work

        # Get the indices of empty rows of B
        nonzero_B_rows = np.where(B.any(axis=1))[0]

        # Form the bare minimum double Toeplitz matrix
        dt_helper = MinimalToeplitzConvolver(input_matrix_shape, kernel)
        cols = nonzero_B_rows
        rows = dt_helper._get_nonzero_rows(cols)
        data = dt_helper._get_values(rows, cols)
        rows = rows.flatten()
        rows_per_col = dt_helper.kernel.size
        cols = np.repeat(cols, rows_per_col)
        DT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=dt_helper.shape)

        # Do the roar
        out_uncropped = DT @ B

        # Unvectorize output
        so = size_output_array = ((input_matrix_shape[0] + k_shape[0] - 1), (kernel.shape[1] + input_matrix_shape[1] -1))
        out_uncropped = np.flip(out_uncropped.reshape(batch_size, so[0], so[1]), axis=1)

        # Crop the output to the correct size
        if mode == 'full':
            t = 0
            b = so[0]+1
            l = 0
            r = so[1]+1
        if mode == 'same':
            t = (kernel.shape[0]-1)//2
            b = -(kernel.shape[0]-1)//2
            l = (kernel.shape[1]-1)//2
            r = -(kernel.shape[1]-1)//2

            b = input_matrix_shape[0]+1 if b==0 else b
            r = input_matrix_shape[1]+1 if r==0 else r
        if mode == 'valid':
            t = (kernel.shape[0]-1)
            l = (kernel.shape[1]-1)
            b = -(kernel.shape[0]-1)
            r = -(kernel.shape[1]-1)

            b = input_matrix_shape[0]+1 if b==0 else b
            r = input_matrix_shape[1]+1 if r==0 else r

        # Crop the output
        output = out_uncropped[:, t:b, l:r]

        return output
    
    def _get_values(self, row_matrix: np.ndarray, col_vector: np.ndarray) -> np.ndarray:
        """
        Compute the values of the matrix at position (row, col) dynamically.
        If any (row,col) corresponding to a zero value in the matrix is
        provided, an exception will be thrown. This is because the
        current program is already designed to only provide coords of
        non-zero values, and we can just save some miniscule amount of
        time by not bothering to make it work for zero values. Or maybe
        it could be done, I unno, I don't think it matters.

        Parameters:
        - row_matrix (np.ndarray): The input row matrix.
        - col_vector (np.ndarray): The input column vector.

        Returns:
        - np.ndarray: The computed values of the matrix at position (row, col).

        Note:
        - The input arrays `row_matrix` and `col_vector` should have compatible shapes.
        - The shape of `row_matrix` should be (C,) or (C, R), where N is the number of columns and K is the number of rows.
        - The shape of `col_vector` should be (C,) where C is the number of columns.
        - The returned array will be a (C*R, 2) array
        """
        # Find which inner Toeplitz this row/col is in,
        # whilst simultaneously retrieiving its toeplitz-relative row/col
        tb_col_vector, t_col_vector = np.divmod(col_vector, self.single_toeplitz_width)
        tb_row_matrix, t_row_matrix = np.divmod(row_matrix, self.single_toeplitz_height)

        # Coordinates of kernel that correspond to the coordinates in row_matrix and col_vector
        kernel_row_matrix = (self.kernel.shape[0] - 1) - (tb_row_matrix - tb_col_vector[:, None])
        del tb_col_vector # Free up memory. TODO: profile to see if this is necessary
        del tb_row_matrix
        padded_kernel_col_matrix = t_row_matrix - t_col_vector[:, None]
        del t_col_vector
        del t_row_matrix
        padded_kernel_indices = np.stack((kernel_row_matrix, padded_kernel_col_matrix), axis=-1).reshape(-1, 2)
        del padded_kernel_col_matrix
        del kernel_row_matrix

        # Return a 1D numpy array of the values of kernel that correspond to the coordinates in padded_kernel_indices
        result = self.kernel[tuple(padded_kernel_indices.T)]

        return result

    def _get_nonzero_rows(self, col_vector: np.ndarray) -> np.ndarray:
        """
        Get the indices of nonzero elements in the columns `col_vector` of the matrix.
        """      
        assert col_vector.ndim == 1
        if col_vector.size == 0:
            return np.array([], dtype=int)
        
        # Compute toeplitz block and relative indices
        tb_col_vector, t_col_vector = np.divmod(col_vector, self.single_toeplitz_width)

        # Compute row offsets for each column
        i_range = self.kernel.shape[0] * self.single_toeplitz_height
        i_offsets_slice = np.arange(self.kernel.shape[1]) + np.arange(0, i_range, self.single_toeplitz_height)[:, None]
        i_offsets_slice = i_offsets_slice.ravel()  # Flatten for efficient broadcasting

        # Compute full row indices
        i_matrix = t_col_vector[:, None] + i_offsets_slice
        i_matrix += tb_col_vector[:, None] * self.single_toeplitz_height

        return i_matrix

    