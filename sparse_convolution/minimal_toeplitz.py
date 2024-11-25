import numpy as np
from typing import Tuple, Optional, Union
import scipy.sparse

class MinimalToeplitzConvolver():
    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
        verbose: Union[bool, int] = False,
    ):
        ## Type checking
        assert isinstance(x_shape, (tuple, list)), f"x_shape must be a tuple. Found: {type(x_shape)}"
        assert all([isinstance(s, (int, float, np.integer, np.floating)) for s in x_shape]), f"x_shape must be a tuple of integers. Found: {[type(s) for s in x_shape]}"
        x_shape = (int(x_shape[0]), int(x_shape[1]))
        assert isinstance(k, np.ndarray), "k must be a numpy array"
        assert k.ndim == 2, "k must be a 2D array"
        assert isinstance(mode, str), "mode must be a string"
        assert mode in ['full', 'same', 'valid'], "mode must be 'full', 'same', or 'valid'"
        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        # New assertions
        assert k.shape[0] > 0 and k.shape[1] > 0, "Kernel must have width and height greater than zero"
        assert x_shape[0] > 0 and x_shape[1] > 0, "Input matrix must have width and height greater than zero"

        self.x_shape: Tuple[int, int] = x_shape
        k = np.flipud(k.copy())
        self.kernel: np.ndarray = k
        self.mode = mode
        self.dtype = k.dtype if dtype is None else dtype # TODO: add dtype to the _get_values method or something so the output is the correct dtype
        self.verbose = verbose

        # Compute some tings
        self.padded_kernel_height: int = x_shape[0] + k.shape[0] - 1
        self.padded_kernel_width: int = x_shape[1] + k.shape[1] - 1
        self.single_toeplitz_height: int = self.padded_kernel_width
        self.single_toeplitz_width: int = x_shape[1]
        self.double_toeplitz_shape: Tuple[int, int] = (self.single_toeplitz_height * self.padded_kernel_height, self.single_toeplitz_width * x_shape[0])

    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Perform a convolution operation between the input matrix and the kernel.
        """
        is_sparse = scipy.sparse.issparse(x)
        if is_sparse:
            x = x.tocsr()
        if batching:
            batch_size = x.shape[0]
        else:
            batch_size = 1
        
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        
        # Make B (the flattened and horizontally stacked input matrices)
        B = x.reshape(batch_size, -1).T
        
        # TODO: Make sure this works with both dense and sparse matrices, with and without batching

        # Get the indices of empty rows of B
        if is_sparse:
            nonzero_B_rows = B.getnnz(axis=1).nonzero()[0]
        else:
            nonzero_B_rows = np.where(B.any(axis=1))[0]

        # Form the bare minimum double Toeplitz matrix
        cols = nonzero_B_rows
        rows = self._get_nonzero_rows(cols)
        data = self._get_values(rows, cols)
        rows = rows.flatten()
        rows_per_col = self.kernel.size
        cols = np.repeat(cols, rows_per_col)
        DT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=self.double_toeplitz_shape) # TODO: add arg dtype=self.dtype (doing later in case it breaks something)

        # Do the roar
        out_uncropped = DT @ B

        # Unvectorize output
        so = size_output_array = ((self.x_shape[0] + self.kernel.shape[0] - 1), (self.kernel.shape[1] + self.x_shape[1] -1))
        # Reconcile it with the original implementation's output shape

        # Crop the output to the correct size
        if mode == 'full':
            t = 0
            b = so[0]+1
            l = 0
            r = so[1]+1
        if mode == 'same':
            t = (self.kernel.shape[0]-1)//2
            b = -(self.kernel.shape[0]-1)//2
            l = (self.kernel.shape[1]-1)//2
            r = -(self.kernel.shape[1]-1)//2

            b = self.x_shape[0]+1 if b==0 else b
            r = self.x_shape[1]+1 if r==0 else r
        if mode == 'valid':
            t = (self.kernel.shape[0]-1)
            l = (self.kernel.shape[1]-1)
            b = -(self.kernel.shape[0]-1)
            r = -(self.kernel.shape[1]-1)

            b = self.x_shape[0]+1 if b==0 else b
            r = self.x_shape[1]+1 if r==0 else r

        if batching:
            idx_crop = np.zeros(so, dtype=np.bool_)
            idx_crop[t:b, l:r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_uncropped[idx_crop,:].T
        else:
            if is_sparse:
                out = out_uncropped.reshape(so).tocsc()[t:b, l:r]
            else:
                out = out_uncropped.reshape(so)[t:b, l:r]  ## reshape back into 2D array and crop
        return out
    
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
         # Free up memory along the way
        del tb_col_vector
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

    