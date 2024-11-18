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
    TEST_ACCURACY = False

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
        (10,10, 2, 2),
        (20,20, 2, 2),
        (1000,1000, 512, 512),
        (2000,2000, 512, 512),
        (3000,3000, 512, 512),
        (4000,4000, 512, 512),
        (5000,5000, 512, 512),
        (6000,6000, 512, 512),
        (7000,7000, 512, 512),
        (8000,8000, 512, 512),
        (9000,9000, 512, 512),
        (10000,10000, 512, 512),
        # (3,3, 2, 2), # TODO: Fix how it fails if the input matrix height is incerased by 1 (increasing the other values by 1 works though)
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

    mode = 'same'
    # mode = 'full'
    # mode = 'valid'

    for batch_size in bstt:
        for density in d_tt:
            for shape in stt:
                print(f'\nTesting shape: {shape}, batch_size: {batch_size}, density: {density}, mode: {mode}')
                # Make X
                X_shape = (batch_size, shape[0], shape[1])
                X = np.zeros(X_shape)
                for i in range(X_shape[0]):
                    X[i] = scipy.sparse.random(X_shape[1], X_shape[2], density=density).toarray() * 100

                x_t = np.flip(X, axis=1).reshape(batch_size, -1).T

                # Make kernel
                k_shape = (shape[2], shape[3])
                kernel = np.random.rand(*k_shape)

                # Form double Toeplitz
                init_start = time.time()
                dt = DoubleToeplitzMatrix(X_shape, k_shape, kernel)
                init_end = time.time()
                print(f'Init time:\t\t{init_end - init_start:.2e}s')

                mul_start = time.time()
                out = dt @ x_t
                mul_end = time.time()
                print(f'Multiplying time:\t{mul_end - mul_start:.2e}s')

                # print(f'DOUBLE TOEPLITZ MATRIX: \n{dt}')

                # Unvectorize output
                so = size_output_array = ((shape[0] + k_shape[0] - 1), (kernel.shape[1] + shape[1] -1))  ## 'size out' is the size of the output array
                out = np.flip(out.reshape(batch_size, so[0], so[1]), axis=1)

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

                    b = shape[0]+1 if b==0 else b
                    r = shape[1]+1 if r==0 else r
                if mode == 'valid':
                    t = (kernel.shape[0]-1)
                    b = -(kernel.shape[0]-1)
                    l = (kernel.shape[1]-1)
                    r = -(kernel.shape[1]-1)

                    b = shape[0]+1 if b==0 else b
                    r = shape[1]+1 if r==0 else r

                # Crop the output
                out = out[:, t:b, l:r]
                # print(f'OUTPUT: \n{out.round(2)}')

                if TEST_ACCURACY:
                    # Compute expected output using convolve2d for each batch
                    expected_output = np.zeros((batch_size, shape[0], shape[1]))
                    for i in range(batch_size):
                        # Apply convolution for each batch element
                        expected_output[i] = convolve2d(X[i], kernel, mode=mode)
                    
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