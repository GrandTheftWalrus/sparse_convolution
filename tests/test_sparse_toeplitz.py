from sparse_convolution import DoubleToeplitzMatrix
import numpy as np
import scipy
import time

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
        (512,512, 32, 32),
        (10000,10000, 1, 1),
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
        0.001,
        # 0.01,
        # 0.1,
        # 1.0
    ]

    for batch_size in bstt:
        for density in d_tt:
            for shape in stt:
                print()
                print(f'input_size: {shape[0]}x{shape[1]} @ density {density}')
                print(f'kernel_size: {shape[2]}x{shape[3]}')
                print(f'batch_size: {batch_size}')

                # Make X
                x_shape = (batch_size, shape[0], shape[1])
                n_samples = shape[0] * shape[1]
                n_nnz = int(n_samples * density)
                x = scipy.sparse.lil_matrix((batch_size, n_samples))
                for i_row in range(x.shape[0]):
                    x[i_row, (np.floor(np.linspace(0, n_samples, n_nnz, endpoint=False)))] = np.random.rand(n_nnz)
                x = x.tocsc()
                x_v = x.toarray().reshape(batch_size, -1)

                # Make kernel
                k_shape = (shape[2], shape[3])
                kernel_values = np.random.rand(*k_shape)
                
                # Form double Toeplitz
                init_start = time.time()
                dt = DoubleToeplitzMatrix(x_shape, k_shape, kernel_values)
                init_end = time.time()
                print(f'DoubleToeplitzMatrix init time: {init_end - init_start}')

                mul_start = time.time()
                out = dt @ x_v
                mul_end = time.time()
                print(f'DoubleToeplitzMatrix mul time: {mul_end - mul_start}')

                assert out.shape == x_shape
                if TEST_ACCURACY:
                    assert np.allclose(out, np.matmul(kernel_values, x.toarray().reshape(batch_size, -1)).reshape(x_shape))