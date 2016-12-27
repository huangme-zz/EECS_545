from sklearn import preprocessing
from numpy.linalg import inv
import numpy as np
from matplotlib import pyplot as plt

def readData(filename, test=False):
  # read data from files
  cols = (1,2,3,4,5,6,7,8,9)
  if test:
    cols = (1,2,3,4,5,6,7,8)
  data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=cols)

  X = data[:, :-1]
  t = data[:, -1]

  if test:
    X = data[:, :]
    t = None

  return X, t

def rmse(predictions, targets):
  # calculate RMSE between predictions and targets
  predictions = np.array(predictions)
  targets = np.array(targets)
  return np.sqrt(((predictions - targets) ** 2).mean())

def getK(X, kernel):
  N, M = X.shape
  K = np.zeros((N, N))
  for i in xrange(N):
    for j in xrange(i, N):
      temp = kernel(X[i, :], X[j, :])
      K[i, j] = temp
      K[j, i] = temp

  return K

def predict(train_X, train_t, test_X, kernel):
  K = getK(train_X, kernel)

  temp = inv(np.identity(K.shape[0]) + K)

  predictions = []
  for i in xrange(test_X.shape[0]):
    k = np.zeros(train_X.shape[0])
    x = test_X[i, :]
    for j in xrange(train_X.shape[0]):
      k[j] = kernel(x, train_X[j, :])

    k = np.matrix(k)
    temp = np.matrix(temp)
    t = np.matrix(train_t).T
    predictions.append((k * temp * t)[0,0])

  return np.array(predictions)



# Part 1
def kernelA(u, v):
  return (u.dot(v) + 1) ** 2

# Part 2
def kernelB(u, v):
  return (u.dot(v) + 1) ** 3

# Part 3
def kernelC(u, v):
  return (u.dot(v) + 1) ** 4

# Part 4
def kernelD(u, v):
  return np.exp(- float(sum((u - v)**2)) / float(2) )



def main():
  train_X, train_t = readData('steel_composition_train.csv')

  scaler = preprocessing.StandardScaler().fit(train_X)
  train_X_std = scaler.transform(train_X)

  kernel_list = [kernelA, kernelB, kernelC, kernelD]
  name_list = ['(i)', '(ii)', '(iii)', '(iv)']

  for name, kernel in zip(name_list, kernel_list):
    predictions = predict(train_X_std, train_t, train_X_std, kernel)
    print "%s the training rmse is: %f" % (name, rmse(predictions, train_t))



if __name__ == "__main__":
  main()