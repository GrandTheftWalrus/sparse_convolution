import numpy as np
import scipy
import scipy.sparse
import time
from sparse_convolution.minimal_toeplitz import MinimalToeplitzConvolver
from sparse_convolution.sparse_convolution import Toeplitz_convolution2d
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
        (1000,1000, 16, 16),
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
        # (100,100, 2, 2),
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
        # (512,512, 32, 32),
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
        2, 
        # 100, 
        # 9999, 
        # 100000
    ]

    d_tt = sparsity_to_try = [
        # 1.0,
        # 0.1,
        # 0.04,
        0.01,
        # 0.005,
        # 0.001,
        # 0.0001,
    ]

    # Note to self: It normally does quite well at
    # 0.001 density with 16x16 kernel and 100x100 input
    # (usually a 7x-10x speedup), but sometimes the
    # speedup is something horrible like 0.035x,
    # for example.
    # AHA! Still no idea why, but I noticed that this
    # happens after having recently done a large
    # image. Perhaps it's something to do with
    # memory allocation?

    # Theory: This program scales like O(s * k^2 * n^2), s < 1
    # whilst conv2d scales like O(k^2 * n^2)

    # I found experimentally that this program begins
    # to outperform convolve2d when the matrix density
    # is less than 0.05. Also, the speedup actually
    # increases as the kernel size increases,
    # pinterestingly, but it eventually levels off.
    # With really small input matrices (< 60 x 60),
    # convolve2d will outperform due to the overhead
    # unless the density is even lower than previously
    # mentioned.

    # Speedup doesn't vary much across input matrix size.
    # Speedup increases with kernel size, but eventually
    # levels off. Speedup is directly proportional to
    # the sparsity of the input matrix.
    # Here are some calculated speedups vs. convolve2d:
    # (100,100, 16, 16) 0.01 density: 2.531x TODO: fill these in
    # (100,100, 16, 16) 0.001 density: 7.241x
    # (1000,1000, 16, 16) 0.01 density: 3.5x
    # (1000,1000, 16, 16) 0.005 density: 4.5x
    # (1000,1000, 16, 16) 0.001 density: 6.5x

    # TODO: Benchmark it against the current implementation
    # TODO: Get the tests set up so that they show it all works
    # and then I can submit a PR

    modes = [
        # 'full',
        'same',
        # 'valid'
        ]

    for batch_size in bs_tt:
        for density in d_tt:
            for shape in stt:
                for mode in modes:
                    for matrix_type in mt_tt:
                        print(f'\nTesting shape: {shape}, batch_size: {batch_size}, density: {density}, mode: {mode}, matrix type: {matrix_type}')
                        # Generate random testing matrices
                        input_matrices_shape = (batch_size, shape[0], shape[1])
                        input_matrices_dense = None
                        input_matrices_sparse = None
                        np.random.seed(0)
                        if batch_size == 1:
                            input_matrices_sparse = scipy.sparse.random(shape[0], shape[1], density=density) * 100
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
                        start_time_old = time.time()
                        conv_new = MinimalToeplitzConvolver(
                            x_shape=shape[:2],
                            k=kernel,
                            mode='same',
                            dtype=np.float32,
                        )

                        # Convolve
                        output_new = conv_new(
                            x=input_matrices_dense,
                            batching=(batch_size > 1),
                        ).toarray()
                        time_taken_new = time.time() - start_time_old

                        # # Test old implementation
                        # start_time_old = time.time()
                        # conv_old = Toeplitz_convolution2d(
                        #     x_shape=shape[:2],
                        #     k=kernel,
                        #     mode='same',
                        #     dtype=np.float32,
                        # )

                        # # Convolve
                        # output_old = conv_new(
                        #     x=input_matrices_dense,
                        #     batching=(batch_size > 1),
                        # ).toarray()
                        # time_taken_old = time.time() - start_time_old

                        # print(f'Old time taken: {time_taken_old:8.2f}s')
                        print(f'New time taken: {time_taken_new:8.2f}s')
                        print(f'Speedup: {time_taken_old / time_taken_new:8.2f}x')

                        if TEST_ACCURACY:
                            # Compute expected output using convolve2d for each batch
                            blank_matrix = np.zeros(input_matrices_shape[1:])
                            expected_output_shape = convolve2d(blank_matrix, kernel, mode=mode).shape
                            expected_output = np.zeros((batch_size, *expected_output_shape))
                            conv2d_start = time.time()
                            for i in range(batch_size):
                                # Apply convolution for each batch element
                                expected_output[i] = convolve2d(input_matrices_dense[i], kernel, mode=mode)
                            conv2d_time = time.time() - conv2d_start

                            assert output_new.shape == expected_output.shape, (
                                f"Output shape mismatch: {output_new.shape} vs {expected_output.shape}"
                            )
                            # Verify each batch's output
                            for i in range(batch_size):
                                assert np.allclose(output_new[i], expected_output[i], atol=1e-6), (
                                    f"Output mismatch for batch index {i}:\n"
                                    f"Input:\n{input_matrices_dense[i]}\n"
                                    f"Kernel:\n{kernel}\n"
                                    f"Expected:\n{expected_output[i]}\n"
                                    f"Got:\n{output_new[i]}\n"
                                )
                            print(
                                # f'init_time:       {init_time:8.3f}s'
                                f'\ntime taken:           {time_taken_new:8.3f}s'
                                f'\nconv2d time:     {conv2d_time:8.3f}s')
                            print(f'Speedup:         {conv2d_time / time_taken_new:8.3f}x')
                        else:
                            print(f'time taken:       {time_taken_new:8.2f}s')

if __name__ == '__main__':
    test_sparse_toeplitz()