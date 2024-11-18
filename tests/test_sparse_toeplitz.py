import numpy as np
import scipy
import time
from sparse_convolution.double_toeplitz_matrix import DoubleToeplitzMatrix
from scipy.signal import convolve2d

def test_sparse_toeplitz():
    """
    Test SparseToeplitzMatrix
    Tests for correct indexing, multiplication, space complexity, and batching.

    VJ 2024
    """

    print(f'testing SparseToeplitzMatrix')
    
    # Warning: Testing the accuracy scales horribly because
    # it checks the output against an explicit matrix
    # multiplication.
    TEST_ACCURACY = True

    stt = shapes_to_try = [
        # (1,1000, 1,1),
        # (1,1000, 1,100),
        # (1,1000, 100,1),
        # (1,1000, 100,100),
        # (1,1,     1,1000),
        # (1,100,   1,1000),
        # (100,1,   1,1000),
        # (1,1000, 1,5),
        # (1,100000, 1,5),
        # (1,1000000, 1,5),
        # (1,10000, 1,2),
        # (1,10000, 1,20),
        # (1,10000, 1,200),
        # (2,2, 10,10),
        # (16,16, 10,10),
        # (64,64, 10,10),
        # (256,256, 10,10),
        # (1024,1024, 10,10),
        (2,3, 2, 2),
        # (512,512, 32, 32),
        # (10000,10000, 1, 1),
        # (10000,10000, 3, 3),
        # (10000,10000, 5, 5),
        # (10000,10000, 3, 3),
        # (10000,10000, 5, 5),
        # (10000,10000, 10, 10),
        #(256,256, 5,20),
        # (256,256, 5,80),
        # (256,256, 2,2),
        # (256,256, 4,4),
        # (256,256, 16,16),
        # (256,256, 64,64),
    ]

    bstt = batch_sizes_to_try = [
        1, 
        # 10, 
        # 100, 
        # 9999, 
        # 100000
    ]

    d_tt = sparsity_to_try = [
        # 0.0001,
        # 0.001,
        0.5,
        # 0.01,
        # 0.1,
        # 1.0
    ]

    for batch_size in bstt:
        for density in d_tt:
            for shape in stt:
                # Make X
                X_shape = (batch_size, shape[0], shape[1])
                X = np.zeros(X_shape)
                for i in range(X_shape[0]):
                    X[i] = scipy.sparse.random(X_shape[1], X_shape[2], density=density).toarray()
                
                # Explicitly  define X for testing
                X = np.array([
                    [
                        [1, 2, 3],
                        [4, 5, 6]
                    ]])
                x_t = X.reshape(batch_size, -1).transpose()

                # Make kernel
                k_shape = (shape[2], shape[3])
                kernel = np.random.rand(*k_shape)

                # Explicitly define kernel for testing
                kernel = np.array([
                    [10, 20],
                    [30, 40]
                ])
                
                # Form double Toeplitz
                init_start = time.time()
                dt = DoubleToeplitzMatrix(X_shape, k_shape, np.flip(kernel))
                init_end = time.time()
                print(f'DoubleToeplitzMatrix init time: {init_end - init_start}')

                mul_start = time.time()
                out = dt @ x_t
                mul_end = time.time()
                print(f'DoubleToeplitzMatrix mul time: {mul_end - mul_start}')

                print(f'DOUBLE TOEPLITZ MATRIX: \n{dt}')

                # Unvectorize output
                out = out.reshape(batch_size, shape[0] + k_shape[0] - 1, shape[1] + k_shape[1] - 1)
                
                # Calculate the padding
                pad_rows = k_shape[0] - 1
                pad_cols = k_shape[1] - 1
                # Handle cropping to remove padding
                crop_start_row = pad_rows // 2
                crop_end_row = -(pad_rows - crop_start_row) if pad_rows > 0 else None

                crop_start_col = pad_cols // 2
                crop_end_col = -(pad_cols - crop_start_col) if pad_cols > 0 else None

                # Crop the output
                out = out[
                    :,  # Batch dimension
                    crop_start_row:crop_end_row,
                    crop_start_col:crop_end_col
                ]

                if TEST_ACCURACY:
                    # Compute expected output using convolve2d for each batch
                    expected_output = np.zeros((batch_size, shape[0], shape[1]))
                    for i in range(batch_size):
                        # Apply convolution for each batch element
                        expected_output[i] = convolve2d(X[i], kernel, mode='same')
                    
                    assert out.shape == expected_output.shape, (
                        f"Output shape mismatch: {out.shape} vs {expected_output.shape}"
                    )
                    # Verify each batch's output
                    for i in range(batch_size):
                        assert np.allclose(out[i], expected_output[i], atol=1e-6), (
                            f"Output mismatch for batch {i}:\n"
                            f"Expected:\n{expected_output[i]}\n"
                            f"Got:\n{out[i]}"
                        )

if __name__ == '__main__':
    test_sparse_toeplitz()