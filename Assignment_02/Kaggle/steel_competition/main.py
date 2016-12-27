from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
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

def main():
  train_X, train_t = readData('steel_composition_train.csv')
  test_X, test_t = readData('steel_composition_test.csv', test=True)
  cols = [0, 3, 5, 6, 7]
  for j in cols:
    for i in xrange(len(train_X)):
      train_X[i][j] = np.log(train_X[i][j])

  alphas = np.logspace(-2, 2, 50)
  ridge_cv = RidgeCV(alphas)
  ridge_cv.fit(train_X, train_t)

  cols = [0, 3, 5, 6, 7]
  for j in cols:
    for i in xrange(len(test_X)):
      test_X[i][j] = np.log(test_X[i][j])

  pred_t = ridge_cv.predict(test_X)

  output_matrix = []
  for (i, t) in zip(range(1, len(pred_t)+1), pred_t):
    output_matrix.append([i, t])
  output_matrix = np.mat(output_matrix)
  np.savetxt('final_result.csv', output_matrix, fmt=('%i', '%.2f'), delimiter=',',\
            header='id,category', comments='')


if __name__ == '__main__':
  main()
