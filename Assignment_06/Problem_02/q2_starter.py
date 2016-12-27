import numpy as np

epsilon = 1e-4
M = 100
K = 50
D = 30
N = 10
nCheck = 1000

def sigmoid(x):
    return 1 / (1+np.exp(-x))


def forwardprop(x, t, A, S, W):
    
    y = 0
    z = 0
    P = 0
    J = 0

    return y, z, P, J


def backprop(x, t, A, S, W):

    y, z, P, J = forwardprop(x, t, A, S, W)
    I = np.zeros((N, 1), dtype=np.float)
    I[t] = 1

    # ---------- make your implementation here -------------
    grad_W = 0
    grad_S = 0
    grad_A = 0
    # -------------------------------------------------

    return grad_A, grad_S, grad_W

def gradient_check():

    A = np.random.rand(K, M+1)*0.1-0.05
    S = np.random.rand(D, K+1)*0.1-0.05
    W = np.random.rand(N, D+1)*0.1-0.05
    x, t = np.random.rand(M, 1)*0.1-0.05, np.random.choice(range(N), 1)[0]

    grad_A, grad_S, grad_W = backprop(x, t, A, S, W)
    errA, errS, errW = [], [], []

    for i in range(nCheck):

        # ---------- make your implementation here -------------
        idx_x, idx_y = 0, 0
        # numerical gradient at (idx_x, idx_y)
        numerical_grad_A = 0
        errA.append(np.abs(grad_A[idx_x, idx_y] - numerical_grad_A))

        idx_x, idx_y = 0, 0
        # numerical gradient at (idx_x, idx_y)
        numerical_grad_S = 0
        errS.append(np.abs(grad_S[idx_x, idx_y] - numerical_grad_S))

        idx_x, idx_y = 0, 0
        # numerical gradient at (idx_x, idx_y)
        numerical_grad_W = 0
        errW.append(np.abs(grad_W[idx_x, idx_y] - numerical_grad_W))
        # -------------------------------------------------

    print 'Gradient checking A, MAE: %.8f' % np.mean(errA)
    print 'Gradient checking S, MAE: %.8f' % np.mean(errS)
    print 'Gradient checking W, MAE: %.8f' % np.mean(errW)


if __name__ == '__main__':
    gradient_check()