from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# Generate the data according to the specification in the homework description
# for part (b)

A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
phi = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi0 = np.array([0.5, 0.3, 0.2])

X = []

for _ in xrange(5000):
    z = [np.random.choice([0,1,2], p=pi0)]
    for _ in range(3):
        z.append(np.random.choice([0,1,2], p=A[z[-1]]))
    x = [np.random.choice([0,1], p=phi[zi]) for zi in z]
    X.append(x)

# TODO: Implement Baum-Welch for estimating the parameters of the HMM
K = 3
N = 4
V = 2
A_origin = A
phi_origin = phi
pi0_origin = pi0


def initialization():
  A = np.random.uniform(0,1,(3,3))
  for i in xrange(3):
    A[i,:] /= np.sum(A[i,:])
  phi = np.random.uniform(0,1,(3,2))
  for i in xrange(3):
    A[i,:] /= np.sum(A[i,:])
  pi0 = np.random.uniform(0,1,3)
  pi0 /= np.sum(pi0)

  return A, phi, pi0

def forwardProcedure(A, phi, pi, observations):
  Alphas = []

  for observation in observations:
    Alpha = np.zeros((K, N))
    for k in xrange(K):
      Alpha[k,0] = pi[k] * phi[k,observation[0]]

    for n in xrange(1, N):
      for j in xrange(K):
        Alpha[j,n] = phi[j,observation[n]]
        temp = 0.0
        for k in xrange(K):
          temp += Alpha[k,n-1] * A[k,j]
        Alpha[j,n] *= temp

    Alphas.append(Alpha)

  return Alphas

def backwardProcedure(A, phi, pi, observations):
  Betas = []

  for observation in observations:
    Beta = np.zeros((K,N))

    for k in xrange(K):
      Beta[k,N-1] = 1.0

    for n in xrange(1, N):
      n = N - n - 1
      for j in xrange(K):
        Beta[j,n] = 0.0
        for k in xrange(K):
          Beta[j,n] += Beta[k,n+1] * phi[k,observation[n+1]] * A[j,k]

    Betas.append(Beta)

  return Betas

def getGamma(Alphas, Betas):
  Gammas = []

  for (Alpha, Beta) in zip(Alphas, Betas):
    Gamma = np.zeros((K,N))
    for n in xrange(N):
      temp = 0.0
      for k in xrange(K):
        temp += Alpha[k,n] * Beta[k,n]
      for j in xrange(K):
        Gamma[j,n] = Alpha[j,n] * Beta[j,n] / temp
    Gammas.append(Gamma)

  return Gammas

def getEta(A, phi, Alphas, Betas, observations):
  Etas = []
  for (Alpha, Beta, observation) in zip(Alphas, Betas, observations):
    Eta = []
    for n_s in xrange(N-1):
      n = N - n_s - 1
      eta = np.zeros((K,K))
      temp = 0.0
      for k in xrange(K):
        temp += Alpha[k,n] * Beta[k,n]
      for i in xrange(K):
        for j in xrange(K):
          eta[i,j] = (Alpha[i,n-1] * A[i,j] * Beta[j,n] * phi[j,observation[n]]) / temp
      Eta.insert(0, eta)
    Etas.append(Eta)

  return Etas


def update(A, phi, pi, Gammas, Etas, observations):
  R = len(observations)

  # update pi
  temp = 0.0
  for r in xrange(R):
    for j in xrange(K):
      temp += Gammas[r][j,0]

  for k in xrange(K):
    pi[k] = 0.0
    for r in xrange(R):
      pi[k] += Gammas[r][k,0]  
    pi[k] /= temp

  # update A
  for j in xrange(K):
    for k in xrange(K):
      A[j,k] = 0.0

      temp = 0.0
      for r in xrange(R):
        for l in xrange(K):
          for n in xrange(N-1):
            temp += Etas[r][n][j,l]

      for r in xrange(R):
        for n in xrange(N-1):
          A[j,k] += Etas[r][n][j,k]

      A[j,k] /= temp

  # update phi
  for k in xrange(K):
    temp = 0.0
    temp1 = 0.0
    for r in xrange(R):
      for n in xrange(N):
        temp += Gammas[r][k,n]
        if observations[r][n] == 1:
          temp1 += Gammas[r][k,n]
    phi[k,1] = temp1 / temp
    phi[k,0] = 1 - phi[k,1]

def distance(A, phi, pi):
  x_v = []
  for i_0 in xrange(2):
    for i_1 in xrange(2):
      for i_2 in xrange(2):
        for i_3 in xrange(2):
          x_v.append((i_0,i_1,i_2,i_3))

  final = []
  for (a0,a1,a2,a3) in x_v:
    s = 0.0
    for i_0 in xrange(3):
      for i_1 in xrange(3):
        for i_2 in xrange(3):
          for i_3 in xrange(3):
            result = pi[i_0] * phi[i_0,a0] * A[i_0,i_1] * phi[i_1,a1] * \
                  A[i_1,i_2] * phi[i_2,a2] * A[i_2,i_3] * phi[i_3,a3]
            s += float(result)

    s_origin = 0.0
    for i_0 in xrange(3):
      for i_1 in xrange(3):
        for i_2 in xrange(3):
          for i_3 in xrange(3):
            result = pi0_origin[i_0] * phi_origin[i_0,a0] * A_origin[i_0,i_1] * \
                     phi_origin[i_1,a1] * A_origin[i_1,i_2] * phi_origin[i_2,a2] \
                     * A_origin[i_2,i_3] * phi_origin[i_3,a3]
            s_origin += float(result)

    final.append(abs(s - s_origin))

  return 0.5 * sum(final)

for R_total in [500, 1000, 2000, 5000]:
  print R_total
  result_list = []
  train_X = X[:R_total]
  A, phi, pi = initialization()
  for _ in xrange(50):
    Alphas = forwardProcedure(A, phi, pi, train_X)
    Betas = backwardProcedure(A, phi, pi, train_X)
    Gammas = getGamma(Alphas, Betas)
    Etas = getEta(A, phi, Alphas, Betas, train_X)
    update(A, phi, pi, Gammas, Etas, train_X)

    result_list.append(distance(A, phi, pi))
  plt.plot(range(50), result_list)
plt.savefig("hw5p3.png")
plt.close()


