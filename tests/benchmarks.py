import traceback
import time

import numpy as np
import scipy.signal

from sparse_convolution import Toeplitz_convolution2d

def benchmark_toeplitz_convolution2d():
    """
    Benchmark toepltiz convolution

    RH 2022
    """

    stt = shapes_to_try = [
        # (1,1000, 1,1),
        # (1,1000, 1,100),
        # (1,1000, 100,1),
        # (1,1000, 100,100),
        # (1,1,     1,1000),
        # (1,100,   1,1000),
        # (100,1,   1,1000),
        (1,1000, 1,5),
        (1,100000, 1,5),
        (1,1000000, 1,5),
        (1,10000, 1,2),
        (1,10000, 1,20),
        (1,10000, 1,200),
        (2,2, 10,10),
        (16,16, 10,10),
        (64,64, 10,10),
        (256,256, 10,10),
        (1024,1024, 10,10),
        (256,256, 5,5),
        (256,256, 5,20),
        (256,256, 5,80),
        (256,256, 2,2),
        (256,256, 4,4),
        (256,256, 16,16),
        (256,256, 64,64),
    ]
    # stt = [s.reshape(-1) for s in stt]

    bstt = batch_sizes_to_try = [
        # 1, 
        # 10, 
        # 100, 
        # 9999, 
        100000
    ]

    d_tt = sparsity_to_try = [
        0.0001,
        # 0.001,
        # 0.01,
        # 0.1,
        # 1.0
    ]

    times = {}

    print(f'testing with batching=False')

    # batching = False
    # for density in d_tt:
    #     for mode in ['full', 'same', 'valid']:
    #         for ii in range(len(stt)):
    #             # x = np.random.rand(stt[ii][0], stt[ii][1])
    #             x = scipy.sparse.rand(stt[ii][0], stt[ii][1], density=density, format='csr').toarray()
    #             k = np.random.rand(stt[ii][2], stt[ii][3])
    #     #         print(stt[0][ii], stt[1][ii], stt[2][ii], stt[3][ii])

    #             try:
    #                 tic_init = time.time()
    #                 t = Toeplitz_convolution2d(x_shape=x.shape, k=k, mode=mode, dtype=None)
    #                 toc_init = time.time()

    #                 tic_t2d = time.time()
    #                 out_t2d = t(x, batching=batching, mode=mode)
    #                 toc_t2d = time.time()

    #                 tic_t2d_s = time.time()
    #                 out_t2d_s = t(scipy.sparse.csr_matrix(x), batching=batching, mode=mode)
    #                 toc_t2d_s = time.time()

    #                 tic_sp = time.time()
    #                 out_sp = scipy.signal.convolve2d(x, k, mode=mode)
    #                 toc_sp = time.time()

    #                 print(f'density: {density}, batching: {batching}  mode: {mode}, init: {toc_init-tic_init:.3f}s, t2d: {toc_t2d-tic_t2d:.3f}s, t2d_s: {toc_t2d_s-tic_t2d_s:.3f}s, sp: {toc_sp-tic_sp:.3f}s,   shapes x,k: {(stt[ii][0], stt[ii][1]), (stt[ii][2], stt[ii][3])}')  
    #                 times[(mode, (stt[ii][0], stt[ii][1]), (stt[ii][2], stt[ii][3]), batching, density)] = (toc_init-tic_init, toc_t2d-tic_t2d, toc_t2d_s-tic_t2d_s, toc_sp-tic_sp)
    #             except Exception as e:
    #                 if mode == 'valid' and (stt[ii][0] < stt[ii][2] or stt[ii][1] < stt[ii][3]):
    #                     if 'x must be larger than k' in str(e):
    #                         continue
    #                 else:
    #                     print(f'A) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
    #                 success = False
    #                 break
    #             try:
    #                 if np.allclose(out_t2d, out_t2d_s.toarray()) and np.allclose(out_t2d, out_sp) and np.allclose(out_sp, out_t2d_s.toarray()):
    #                     success = True
    #                     continue
    #             except Exception as e:
    #                 print(f'B) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
    #                 success = False
    #                 break

    #             else:
    #                 print(f'C) test failed with batching==False, shapes:  x: {x.shape}, k: {k.shape} and mode: {mode}')
    #                 success = False
    #                 break     


    print(f'testing with batching=True')

    batching = True
    for ii in range(len(stt)):
        for density in d_tt:
            for bs in bstt:
                # x = np.stack([np.random.rand(stt[ii][0], stt[ii][1]).reshape(-1) for jj in range(bs)], axis=0)
                # x = scipy.sparse.rand(bs, stt[ii][0] * stt[ii][1], density=density, format='csc')
                n_samples = stt[ii][0] * stt[ii][1]
                n_nnz = int(n_samples * density)
                x = scipy.sparse.lil_matrix((bs, n_samples))
                for i_row in range(x.shape[0]):
                    x[i_row, (np.floor(np.linspace(0, n_samples, n_nnz, endpoint=False)))] = np.random.rand(n_nnz)
                x = x.tocsc()

                k = np.random.rand(stt[ii][2], stt[ii][3])
        #         print(stt[0][ii], stt[1][ii], stt[2][ii], stt[3][ii])
                # print(x.shape, k.shape)
                
                for mode in [
                    'full',
                    # 'same', 
                    # 'valid'
                ]:
                    # try:
                        tic_init = time.time()
                        t = Toeplitz_convolution2d(x_shape=(stt[ii][0], stt[ii][1]), k=k, mode=mode, dtype=None)
                        toc_init = time.time()

                        # x_s = scipy.sparse.csr_matrix(x)
                        tic_t2d_s = time.time()
                        out_t2d_s = t(x, batching=batching, mode=mode)
                        toc_t2d_s = time.time()
                        # out_t2d_s = out_t2d_s.todense()

                        tic_t2d = time.time()
                        # out_t2d = t(x, batching=batching, mode=mode).todense()
                        out_t2d = out_t2d_s
                        toc_t2d = time.time()

                        tic_sp = time.time()
                        # out_sp = np.stack([scipy.signal.convolve2d(x_i.reshape(stt[ii][0], stt[ii][1]), k, mode=mode) for x_i in x], axis=0)
                        out_sp = out_t2d
                        toc_sp = time.time()

                        print(f'init: {toc_init-tic_init:.3f}s, t2d: {toc_t2d-tic_t2d:.3f}s, t2d_s: {toc_t2d_s-tic_t2d_s:.3f}s, sp: {toc_sp-tic_sp:.3f}s,  density: {density}, batching: {bs}  mode: {mode}, shapes x,k: {(stt[ii][0], stt[ii][1]), (stt[ii][2], stt[ii][3])}')  
                        times[(mode, (stt[ii][0], stt[ii][1]), (stt[ii][2], stt[ii][3]), bs, density)] = (toc_init-tic_init, toc_t2d-tic_t2d, toc_t2d_s-tic_t2d_s, toc_sp-tic_sp)
        
                    # except Exception as e:
                    #     if mode == 'valid' and (stt[ii][0] < stt[ii][2] or stt[ii][1] < stt[ii][3]):
                    #         if 'x must be larger than k' in str(e):
                    #             continue
                    #     else:
                    #         print(f'A) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                    #     success = False
                    #     break
                    # try:
                    #     if np.allclose(out_t2d, out_t2d_s) and np.allclose(out_t2d, out_sp) and np.allclose(out_sp, out_t2d_s):
                    #         success = True
                    #         continue
                    # except Exception as e:
                    #     print(f'B) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                    #     success = False
                    #     break

                    # else:
                    #     print(f'C) test failed with batching==False, shapes:  x: {x.shape}, k: {k.shape} and mode: {mode}')
                    #     success = False
                    #     break           

    return times