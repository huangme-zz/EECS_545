import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import math, csv

def readData(filename):
  in_f = open(filename, 'rb')
  csv_file = csv.reader(in_f)

  X = []
  t = []
  i = 0
  for row in csv_file:
    if i == 0:
      i += 1
      continue
    X.append([float(elt) for elt in row[1:-1]])
    t.append(float(row[-1]))

  in_f.close()

  return X, t

def getFeatureMatrix(X, M):
  # create feature matrix
  feature_matrix = []
  for x in X:
    feature_vector = [1.0]
    for entry in x:
      for m in range(1, M+1):
        feature_vector.append(entry ** m)
    feature_matrix.append(feature_vector)
  return feature_matrix

def rmse(predictions, targets):
  # calculate RMSE between predictions and targets
  predictions = np.array(predictions)
  targets = np.array(targets)
  return np.sqrt(((predictions - targets) ** 2).mean())


''' Problem 1.(a)i '''
def partA():
  print 'partA() start...'
  # read training and testing data from files
  train_X, train_t = readData('train_graphs_f16_autopilot_cruise.csv')
  test_X, test_t = readData('test_graphs_f16_autopilot_cruise.csv')

  # initialize parameters
  list_M = range(1, 7)
  K = 6

  list_train_error = []
  list_test_error = []

  # Psuedo-inverse approach for linear regression
  for M in list_M:
    # get Phi of train data
    feature_matrix = getFeatureMatrix(train_X, M)
    Phi = np.mat(feature_matrix)
    t_vector = np.mat(train_t).T

    # get w*
    left = Phi.T * Phi
    right = Phi.T * t_vector

    w = np.linalg.solve(left, right)

    # calculate train error
    predictions = Phi * w
    predictions = np.squeeze(np.asarray(predictions))

    train_error = rmse(predictions, train_t)
    list_train_error.append(train_error)

    # get Phi of test data
    feature_matrix = getFeatureMatrix(test_X, M)
    Phi = np.mat(feature_matrix)

    # calculate test error
    predictions = Phi * w
    predictions = np.squeeze(np.asarray(predictions))

    test_error = rmse(predictions, test_t)
    list_test_error.append(test_error)

  # Plot
  fig = plt.figure()
  plt.plot(list_M, list_train_error, '-og')
  plt.plot(list_M, list_test_error, '-or')
  plt.xlabel('M')
  plt.ylabel('RMSE')
  fig.savefig('1.png')

  print 'partA() end'

  


''' Problem 1.(a)ii '''
def partB():
  print 'partB() start...'
  # read training and testing data from files
  train_X, train_t = readData('train_graphs_f16_autopilot_cruise.csv')
  test_X, test_t = readData('test_graphs_f16_autopilot_cruise.csv')

  # initialize parameters
  list_ln_lambda = range(-40, 21)
  M = 6

  list_train_error = []
  list_test_error = []

  # Psuedo-inverse approach for regularized linear regression
  for ln_lambda in list_ln_lambda:
    # get Phi of train data
    feature_matrix = getFeatureMatrix(train_X, M)
    Phi = np.mat(feature_matrix)
    t_vector = np.mat(train_t).T

    # get w*
    left = Phi.T * Phi + np.exp(ln_lambda) * np.identity(Phi.shape[1])
    right = Phi.T * t_vector

    w = np.linalg.solve(left, right)

    # calculate train error
    predictions = Phi * w
    predictions = np.squeeze(np.asarray(predictions))

    train_error = rmse(predictions, train_t)
    list_train_error.append(train_error)

    # get Phi of test data
    feature_matrix = getFeatureMatrix(test_X, M)
    Phi = np.mat(feature_matrix)

    # calculate test error
    predictions = Phi * w
    predictions = np.squeeze(np.asarray(predictions))

    test_error = rmse(predictions, test_t)
    list_test_error.append(test_error)

  # Plot
  fig = plt.figure()
  plt.plot(list_ln_lambda, list_train_error, '-og')
  plt.plot(list_ln_lambda, list_test_error, '-or')
  plt.xlabel('ln_lambda')
  plt.ylabel('RMSE')
  fig.savefig('2.png')

  print 'partB() end'



''' Problem 1.(b) '''
def partC():
  print 'partC() start...'
  def gausssian(mu, tau, data):
    mu = np.mat(mu)
    data = np.mat(data)

    diff = data - mu
    d = float(diff * diff.T)
    return np.exp(-1.0 * d / (2 * (tau ** 2)))

  def getWeight(mu, tau, X):
    list_r = []
    for x in X:
      r = gausssian(mu, tau, x)
      list_r.append(r)
    return np.diag(list_r)

  # read data from files
  train_X, train_t = readData('train_graphs_f16_autopilot_cruise.csv')
  test_X, test_t = readData('test_locreg_f16_autopilot_cruise.csv')

  # initialize
  list_tau = np.logspace(-2, 1, num=10, base=2)
  M = 1

  # standardization
  scaler = preprocessing.StandardScaler().fit(train_X + test_X)
  train_X_std = scaler.transform(train_X)
  test_X_std = scaler.transform(test_X)

  list_test_error = []

  # Psuedo-inverse approach for regularized linear regression

  # get Phi of training data
  feature_matrix = getFeatureMatrix(train_X_std, M)
  Phi = np.mat(feature_matrix)
  t_vector = np.mat(train_t).T

  for tau in list_tau:
    print tau
    predictions = []
    for (x_test, t_test) in zip(test_X_std, test_t):
      # get the weight of x_test with each training data
      R = getWeight(x_test, tau, train_X_std)

      # get w*
      left = Phi.T * R * Phi
      right = Phi.T * R * t_vector
      w = np.linalg.solve(left, right)

      y = np.mat([1.0] + list(x_test)) * w
      predictions.append(y)

    test_error = rmse(predictions, test_t)
    print test_error
    list_test_error.append(test_error)

  # Plot
  fig = plt.figure()
  plt.plot(list_tau, list_test_error, '-or')
  plt.xlabel('tau')
  plt.ylabel('RMSE')
  fig.savefig('3.png')

  print 'partC() end'



def main():
  # partA()
  # partB()
  partC()

if __name__ == '__main__':
  main()