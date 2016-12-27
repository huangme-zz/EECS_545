from __future__ import division
from matplotlib import pyplot as plt
import numpy as np

# Generate the data according to the specification in the homework description

N = 10000

# Here's an estimate of gamma for you
G = lambda x: np.log(np.cosh(x))
gamma = np.mean(G(np.random.randn(10**6)))

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate((s1.reshape((1,N)), s2.reshape((1,N))), 0)

A = np.array([[1,2],[-2,1]])

X = A.dot(S)

# TODO: Implement ICA using a 2x2 rotation matrix on a whitened version of X
U, Sigma, V = np.linalg.svd(X, full_matrices=False)
Sigma = np.diag(Sigma)
D = np.sqrt(N) * np.linalg.inv(Sigma).dot(U.T)

X_white = D.dot(X)

W = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

theta = 0.0
max_theta = theta
max_J = 0
theta_list = []
result_list = []
while theta <= np.pi/2:
  Y = W(theta).dot(X_white)
  mean_1 = np.mean(G(Y[0,:]))
  mean_2 = np.mean(G(Y[1,:]))
  J = (mean_1 - gamma)**2 + (mean_2 - gamma)**2
  if J > max_J:
    max_J = J
    max_theta = theta
  result_list.append(J)
  theta_list.append(theta)
  theta += np.pi/2000

fig = plt.figure()
plt.plot(theta_list, result_list)
fig.savefig('hw5p5_1.png')

Y = W(max_theta).dot(X_white)
fig2 = plt.figure()
plt.plot(range(N), Y[0,:])
fig2.savefig("hw5p5_2.png")

fig3 = plt.figure()
plt.plot(range(N), Y[1,:])
fig3.savefig("hw5p5_3.png")



