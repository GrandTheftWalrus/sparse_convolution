import numpy as np
import scipy
import scipy.sparse
import time
from sparse_convolution.double_toeplitz_matrix import DoubleToeplitzHelper
from scipy.signal import convolve2d
import pdb

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
        (3,3, 2, 2),
        (10,10, 2, 2),
        (20,20, 2, 2),
        (30,30, 2, 2),
        (40,40, 2, 2),
        (50,50, 2, 2),
        (60,60, 2, 2),
        (70,70, 2, 2),
        (80,80, 2, 2),
        (90,90, 2, 2),
        (100,100, 2, 2),
        (1000,1000, 3, 3),
        (2000,2000, 3, 3),
        (3000,3000, 3, 3),
        (4000,4000, 3, 3),
        (5000,5000, 3, 3),
        (6000,6000, 3, 3),
        (7000,7000, 3, 3),
        (8000,8000, 3, 3),
        (9000,9000, 3, 3),
        (10000,10000, 3, 3),
        # (3,3, 2, 2),
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
        # 0.5,
        0.01,
        # 0.1,
        # 1.0
    ]

    modes = [
        # 'full',
        'same',
        # 'valid'
        ]

    for batch_size in bstt:
        for density in d_tt:
            for shape in stt:
                for mode in modes:
                    print(f'\nTesting shape: {shape}, batch_size: {batch_size}, density: {density}, mode: {mode}')
                    
                    # Generate random testing matrices
                    input_matrices_shape = (batch_size, shape[0], shape[1])
                    input_matrices = np.zeros(input_matrices_shape)
                    for i in range(input_matrices_shape[0]):
                        input_matrices[i] = scipy.sparse.random(input_matrices_shape[1], input_matrices_shape[2], density=density).toarray() * 100
                    
                    # Make B (the flattened and horizontally stacked input matrices)
                    B = np.flip(input_matrices, axis=1).reshape(batch_size, -1).T

                    # Make kernel
                    k_shape = (shape[2], shape[3])
                    kernel = np.random.rand(*k_shape)

                    # Get the indices of empty rows of B
                    nonzero_detection_start = time.time()
                    nonzero_B_rows = np.where(B.any(axis=1))[0].tolist()
                    nonzero_detection_time = time.time() - nonzero_detection_start
                    print(f'nonzero_detection_time: {nonzero_detection_time:8.2f}s')

                    # Form the bare minimum double Toeplitz
                    init_start = time.time()
                    dt_helper = DoubleToeplitzHelper(shape, kernel)
                    num_elements = len(nonzero_B_rows) * k_shape[0] * k_shape[1]
                    rows = np.empty(num_elements, dtype=int)
                    cols = np.empty(num_elements, dtype=int)
                    data = np.empty(num_elements, dtype=int)
                    total_nonzero_row_detection_time = 0
                    total_value_calculation_time = 0
                    current_index = 0
                    for n, col in enumerate(nonzero_B_rows):
                        nz_row_detect_start = time.time()
                        col_rows = dt_helper.get_nonzero_rows(col)
                        num_rows = len(col_rows)
                        total_nonzero_row_detection_time += time.time() - nz_row_detect_start
                        value_calc_start = time.time()
                        # Assign values to the preallocated arrays
                        rows[current_index:current_index + num_rows] = col_rows
                        cols[current_index:current_index + num_rows] = col
                        data[current_index:current_index + num_rows] = [
                            dt_helper.get_value(row, col) for row in col_rows
                        ]
                        current_index += num_rows
                        total_value_calculation_time += time.time() - value_calc_start

                    csr_creation_start = time.time()
                    DT = scipy.sparse.csr_matrix((data, (rows, cols)), shape=dt_helper.shape)
                    csr_creation_time = time.time() - csr_creation_start
                    init_time = time.time() - init_start
                    print(f'total_nonzero_row_detection_time: {total_nonzero_row_detection_time:8.2f}s')
                    print(f'total_value_calculation_time: {total_value_calculation_time:8.2f}s')
                    print(f'csr_creation_time: {csr_creation_time:8.2f}s')

                    mul_start = time.time()
                    out_uncropped = DT @ B
                    mul_time = time.time() - mul_start

                    # Unvectorize output
                    so = size_output_array = ((shape[0] + k_shape[0] - 1), (kernel.shape[1] + shape[1] -1))  ## 'size out' is the size of the output array
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

                        b = shape[0]+1 if b==0 else b
                        r = shape[1]+1 if r==0 else r
                    if mode == 'valid':
                        t = (kernel.shape[0]-1)  # 2 - 1 = 1
                        l = (kernel.shape[1]-1)  # 2 - 1 = 1
                        b = -(kernel.shape[0]-1) # 2 - 1 = -1
                        r = -(kernel.shape[1]-1) # 2 - 1 = -1

                        b = shape[0]+1 if b==0 else b
                        r = shape[1]+1 if r==0 else r

                    # Crop the output
                    out = out_uncropped[:, t:b, l:r]

                    if TEST_ACCURACY:
                        # Compute expected output using convolve2d for each batch
                        blank_matrix = np.zeros(input_matrices_shape[1:])
                        expected_output_shape = convolve2d(blank_matrix, kernel, mode=mode).shape
                        expected_output = np.zeros((batch_size, *expected_output_shape))
                        mul_conv2d_start = time.time()
                        for i in range(batch_size):
                            # Apply convolution for each batch element
                            expected_output[i] = convolve2d(input_matrices[i], kernel, mode=mode)
                        mul_conv2d_time = time.time() - mul_conv2d_start

                        assert out.shape == expected_output.shape, (
                            f"Output shape mismatch: {out.shape} vs {expected_output.shape}"
                        )
                        # Verify each batch's output
                        for i in range(batch_size):
                            assert np.allclose(out[i], expected_output[i], atol=1e-6), (
                                f"Output mismatch for batch index {i}:\n"
                                f"Input:\n{input_matrices[i]}\n"
                                f"Kernel:\n{kernel}\n"
                                f"Expected:\n{expected_output[i]}\n"
                                f"Got:\n{out[i]}\n"
                            )
                        print(f'init_time:       {init_time:8.3f}s'
                            f'\nmul_time:        {mul_time:8.3f}s'
                            f'\ntotal:           {init_time + mul_time:8.3f}s'
                            f'\nmul_conv2d_time: {mul_conv2d_time:8.3f}s')
                        print(f'Speedup:         {mul_conv2d_time / (mul_time + init_time):8.3f}x')
                    else:
                        print(f'init_time:       {init_time:8.2f}s'
                            f'\nmul_time:        {mul_time:8.2f}s'
                            f'\ntotal:           {init_time + mul_time:8.2f}s')

if __name__ == '__main__':
    test_sparse_toeplitz()