import numpy as np
import scipy
import scipy.sparse
import time
from sparse_convolution.old_version import Toeplitz_convolution2d_Old
from sparse_convolution.sparse_convolution import Toeplitz_convolution2d as Toeplitz_convolution2d_New_Version
from scipy.signal import convolve2d
from typing import Dict

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
    TEST_AGAINST_CONV2D = True
    TEST_AGAINST_OLD_VERSION = True
    WRITE_RESULTS = False

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
        # (2,3, 2, 2),
        # (3,3, 2, 2),
        # (10,10, 16, 16),
        # (20,20, 16, 16),
        # (30,30, 16, 16),
        # (40,40, 16, 16),
        # (50,50, 16, 16),
        # (60,60, 16, 16),
        # (70,70, 16, 16),
        # (80,80, 16, 16),
        # (90,90, 16, 16),
        # (100,100, 5, 5),
        # (100,100, 16, 16),
        # (100,100, 16, 16),
        # (100,100, 32, 32),
        # (200,200, 16, 16),
        # (300,300, 16, 16),
        # (400,400, 16, 16),
        # (500,500, 16, 16),
        # (600,600, 16, 16),
        # (700,700, 16, 16),
        # (800,800, 16, 16),
        # (900,900, 16, 16),
        # (1000,1000, 5, 5),
        # (1000,1000, 16, 16),
        # (1000,1000, 32, 32),
        # (10000,10000, 5, 5),
        # (10000,10000, 16, 16),
        # (10000,10000, 32, 32),
        # (2000,2000, 16, 16),
        # (3000,3000, 16, 16),
        # (1, 1, 1, 1),
        # (1000,1000, 1, 1),
        # (1000,1000, 2, 2),
        # (1000,1000, 4, 4),
        # (1000,1000, 9, 9),
        # (1000,1000, 16, 16),
        # (1000,1000, 32, 32),
        # (1000,1000, 64, 64),
        # (1000,1000, 128, 128),
        # (1000,1000, 256, 256),
        (100,100, 2, 2),
        (100,100, 10, 10),
        (50,150, 5, 5),
        (100,1, 5, 1),
        # (1000,1000, 1, 1),
        # (1000,1000, 2, 2),
        # (1000,1000, 3, 3),
        # (1000,1000, 4, 4),
        # (1000,1000, 5, 5),
        # (1000,1000, 6, 6),
        # (1000,1000, 7, 7),
        # (1000,1000, 8, 8),
        # (1000,1000, 9, 9),
        # (1000,1000, 10, 10),
        # (1000,1000, 16, 16),
        # (1000,1000, 32, 32),
        # (1000,1000, 32, 32),
        # (1000,1000, 64, 64),
        # (1000,1000, 40, 40),
        # (1000,1000, 100, 100),
        # (2000,2000, 5, 5),
        # (3000,3000, 3, 3),
        # (4000,4000, 3, 3),
        # (5000,5000, 32, 32),
        # (6000,6000, 3, 3),
        # (7000,7000, 3, 3),
        # (8000,8000, 3, 3),
        # (9000,9000, 3, 3),
        # (10000,10000, 32, 32),
        # (3,3, 2, 2),
        (512,512, 32, 32),
        (1000,1000, 3, 3),
        # (2000,2000, 3, 3),
        # (3000,3000, 3, 3),
        (4000,4000, 3, 3),
        # (5000,5000, 3, 3),
        # (6000,6000, 3, 3),
        # (7000,7000, 3, 3),
        # (8000,8000, 3, 3),
        # (9000,9000, 3, 3),
        # (2000,2000, 3, 3),
        # (2000,2000, 3, 3),
        # (10000,10000, 1, 1),
        # (10000,10000, 3, 3),
        # (10000,10000, 5, 5),
        # (10000,10000, 3, 3),
        # (10000,10000, 5, 5),
        # (10000,10000, 10, 10),
        # (256,256, 5,20),
        # (256,256, 5,80),
        # (256,256, 2,2),
        # (256,256, 4,4),
        # (256,256, 16,16),
        # (256,256, 64,64),
    ]

    mt_tt = matrix_types_to_try = [
        'dense',
        'sparse',
    ]

    bs_tt = batch_sizes_to_try = [
        1, 
        5,
        # 10,
        # 100, 
        # 9999, 
        # 100000
    ]

    d_tt = sparsity_to_try = [
        # 1.0,
        # 0.5,
        # 0.04,
        0.01,
        # 0.005,
        # 0.001,
        0.0001,
    ]

    modes = [
        # 'full',
        'same',
        # 'valid'
        ]
    
    dtypes_tt = [
        # np.int32, # I think this doesn't work in the previous version. It seems to output a bunch of zeros
        np.float32,
    ]

    # Make a statistics matrix with one dimension for each test with more than one value
    num_methods = 1 + TEST_AGAINST_OLD_VERSION + TEST_AGAINST_CONV2D
    num_tests = len(bs_tt) * len(d_tt) * len(stt) * len(mt_tt) * len(modes) * len(dtypes_tt)
    test_results = np.zeros((num_methods, len(bs_tt), len(d_tt), len(stt), len(mt_tt))) # excluding mode

    # I found experimentally that this program begins
    # to outperform convolve2d when the matrix density
    # is less than 0.05. Also, the speedup actually
    # increases as the kernel size increases,
    # pinterestingly, but it eventually levels off.
    # With really small input matrices (< 60 x 60),
    # convolve2d will outperform due to the overhead
    # unless the density is even lower than previously
    # mentioned.

    test_num = 0
    total_time_start = time.time()
    for batch_size in bs_tt:
        for density in d_tt:
            for shape in stt:
                for mode in modes:
                    for matrix_type in mt_tt:
                        for dtype in dtypes_tt:
                            test_num += 1
                            print(f'\nTesting shape: {shape}, batch_size: {batch_size}, density: {density}, mode: {mode}, matrix type: {matrix_type} ({test_num}/{num_tests})')
                            # Generate random testing matrices
                            input_matrices_shape = (batch_size, shape[0], shape[1])
                            input_matrices_dense = None
                            input_matrices_sparse = None
                            np.random.seed(0)
                            if batch_size == 1:
                                input_matrices_sparse = scipy.sparse.random(shape[0], shape[1], density=density) * 100
                                # convert its type to int32
                                if dtype == np.int32:
                                    input_matrices_sparse = input_matrices_sparse.astype(np.int32)
                                input_matrices_dense = input_matrices_sparse.toarray()
                            else:
                                input_matrices_dense = np.zeros(input_matrices_shape)
                                input_matrices_sparse = []
                                for i in range(batch_size):
                                    input_matrices_sparse.append(scipy.sparse.random(shape[0], shape[1], density=density) * 100)
                                    input_matrices_dense[i] = input_matrices_sparse[i].toarray()
                                # Vertically stack sparse matrices
                                input_matrices_sparse = scipy.sparse.vstack([_.reshape(1, -1) for _ in input_matrices_sparse]).tocsr()
                            
                            # Make dense kernel
                            k_shape = (shape[2], shape[3])
                            kernel = np.random.rand(*k_shape)

                            # Test new implementation
                            input_matrices = input_matrices_sparse if matrix_type == 'sparse' else input_matrices_dense
                            stats_new = _test_implementation(Toeplitz_convolution2d_New_Version, input_matrices, shape, kernel, mode, matrix_type, batch_size, dtype)
                            output_new = stats_new['output']
                            raw_output_new = stats_new['raw_output']
                            time_taken_new = stats_new['time_taken']
                            print(f'Time taken (new):\t{time_taken_new:15.2f}s')
                            test_results[0, bs_tt.index(batch_size), d_tt.index(density), stt.index(shape), mt_tt.index(matrix_type)] = time_taken_new

                            # Test old implementation
                            if TEST_AGAINST_OLD_VERSION:
                                input_matrices = input_matrices_sparse if matrix_type == 'sparse' else input_matrices_dense
                                stats_old = _test_implementation(Toeplitz_convolution2d_Old, input_matrices, shape, kernel, mode, matrix_type, batch_size, dtype)
                                output_old = stats_old['output']
                                raw_output_old = stats_old['raw_output']
                                time_taken_old = stats_old['time_taken']
                                test_results[1, bs_tt.index(batch_size), d_tt.index(density), stt.index(shape), mt_tt.index(matrix_type)] = time_taken_old
                                print(f'Time taken (old):\t{time_taken_old:15.2f}s')
                                print(f'Speedup vs. old:    {time_taken_old / time_taken_new:8.2f}x')

                                # Compare outputs
                                assert raw_output_new.shape == raw_output_old.shape, (
                                    f"Raw output shape mismatch: {raw_output_new.shape} vs {raw_output_old.shape}"
                                )
                                assert output_new.shape == output_old.shape, (
                                    f"Reshaped output shape mismatch: {output_new.shape} vs {output_old.shape}"
                                )
                                assert np.allclose(raw_output_new, raw_output_old, atol=1e-6), (
                                    f"Raw output mismatch:\n"
                                    f"Input:\n{input_matrices_dense}\n"
                                    f"Kernel:\n{kernel}\n"
                                    f"Expected:\n{raw_output_old}\n"
                                    f"Got:\n{raw_output_new}\n"
                                )
                                assert np.allclose(output_new, output_old, atol=1e-6), (
                                    f"Output mismatch:\n"
                                    f"Input:\n{input_matrices_dense}\n"
                                    f"Kernel:\n{kernel}\n"
                                    f"Expected:\n{output_old}\n"
                                    f"Got:\n{output_new}\n"
                                )

                            if TEST_AGAINST_CONV2D:
                                # Compute expected output using convolve2d for each batch
                                blank_matrix = np.zeros(shape[:2])
                                expected_output_shape = convolve2d(blank_matrix, kernel, mode=mode).shape
                                expected_output = None
                                conv2d_start = time.time()
                                if batch_size > 1:
                                    expected_output = np.zeros((batch_size, *expected_output_shape))
                                for i in range(batch_size):
                                    # Apply convolution for each batch element
                                    if batch_size == 1:
                                        result = convolve2d(input_matrices_dense, kernel, mode=mode)
                                        if result.dtype != dtype:
                                            result = result.astype(dtype)
                                        expected_output = result
                                    else:
                                        result = convolve2d(input_matrices_dense[i], kernel, mode=mode)
                                        if result.dtype != dtype:
                                            result = result.astype(dtype)
                                        expected_output[i] = result
                                conv2d_time = time.time() - conv2d_start

                                assert output_new.shape == expected_output.shape, (
                                    f"Output shape mismatch: {output_new.shape} vs {expected_output.shape}"
                                )
                                # Verify each batch's output
                                for i in range(batch_size):
                                    if batch_size == 1:
                                        assert np.allclose(output_new, expected_output, atol=1e-6), (
                                            f"Output mismatch:\n"
                                            f"Input:\n{input_matrices_dense}\n"
                                            f"Kernel:\n{kernel}\n"
                                            f"Expected:\n{expected_output}\n"
                                            f"Got:\n{output_new}\n"
                                        )
                                    else:
                                        assert np.allclose(output_new[i], expected_output[i], atol=1e-6), (
                                            f"Output mismatch for batch index {i}:\n"
                                            f"Input:\n{input_matrices_dense[i]}\n"
                                            f"Kernel:\n{kernel}\n"
                                            f"Expected:\n{expected_output[i]}\n"
                                            f"Got:\n{output_new[i]}\n"
                                        )
                                print(f'Conv2d time:        \t{conv2d_time:15.2f}s')
                                print(f'Speedup vs. conv2d: {conv2d_time / time_taken_new:8.2f}x')
                                test_index = 2 if TEST_AGAINST_OLD_VERSION else 1
                                test_results[test_index, bs_tt.index(batch_size), d_tt.index(density), stt.index(shape), mt_tt.index(matrix_type)] = conv2d_time
    total_time_taken = time.time() - total_time_start
    print(f'Total time taken: {total_time_taken:.2f}s')

    if WRITE_RESULTS:
        # Write the stats as a CSV, with column names
        methods = ['updated']
        if TEST_AGAINST_OLD_VERSION:
            methods.append('original')
        if TEST_AGAINST_CONV2D:
            methods.append('conv2d')
        with open('test_results.csv', 'w') as f:
            f.write('method,batch_size,density,shape,matrix_type,time_seconds,memory_usage_bytes\n')
            for method in range(num_methods):
                for bs in bs_tt:
                    for d in d_tt:
                        for s in stt:
                            for mt in mt_tt:
                                time_taken = test_results[method, bs_tt.index(bs), d_tt.index(d), stt.index(s), mt_tt.index(mt), 0]
                                memory_used = test_results[method, bs_tt.index(bs), d_tt.index(d), stt.index(s), mt_tt.index(mt), 1]
                                s = f'{s[0]}x{s[1]}_{s[2]}x{s[3]}'
                                f.write(f'{methods[method]},{bs},{d},{s},{mt},{time_taken:.4f},{memory_used:.0f}\n')

def _test_implementation(conv_function, x, shape, kernel, mode, matrix_type, batch_size, dtype) -> Dict:
    # Get expected dimensions based on mode, for reshaping
    output_shape = None
    if mode == 'full':
        output_shape = (shape[0] + kernel.shape[0] - 1, shape[1] + kernel.shape[1] - 1)
    elif mode == 'same':
        output_shape = (shape[0], shape[1])
    elif mode == 'valid':
        output_shape = (shape[0] - kernel.shape[0] + 1, shape[1] - kernel.shape[1] + 1)
    

    # Test implementation
    start_time = time.time()
    raw_output = None
 
    # Initialize convolution object
    conv = conv_function(
        x_shape=shape[:2],
        k=kernel,
        mode=mode,
        dtype=dtype
    )

    # Convolve
    raw_output = conv(
        x=x.reshape(batch_size, -1, order='C'),
        batching=(batch_size > 1),
    )
    time_taken = time.time() - start_time

    # Reshape output
    if matrix_type == 'sparse':
        raw_output = raw_output.toarray()
    if batch_size == 1:
        reshaped_output = raw_output.copy().reshape(output_shape[0], output_shape[1], order='C')
    else:
        reshaped_output = raw_output.copy().reshape(batch_size, output_shape[0], output_shape[1], order='C')

    return {
        'output': reshaped_output,
        'raw_output': raw_output,
        'time_taken': time_taken,
    }

if __name__ == '__main__':
    test_sparse_toeplitz()