from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def readData(filename):
  # read data from files
  cols = (i for i in xrange(1,785))
  data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=cols)

  return data

def get_patches(X):
  m,n = X.shape
  X = np.pad(X, ((2, 2), (2, 2)), 'constant')
  patches = np.zeros((m*n, 25))
  for i in range(m):
    for j in range(n):
      patches[i*n+j] = X[i:i+5,j:j+5].reshape(25)
  return patches

def main():
  # read data
  train_clean = readData('train_clean.csv')
  train_noised = readData('train_noised.csv')
  test_noised = readData('test_noised.csv')

  # denoising
  patches = get_patches(train_noised)
  patches_test = get_patches(test_noised)
  m, n = train_clean.shape
  train_clean_v = train_clean.reshape(m*n)
  m, n = test_noised.shape

  clf = BaggingRegressor()

  clf.fit(patches, train_clean_v)
  test_clean = clf.predict(patches_test)
  # print mean_squared_error(train_clean_v, test_clean) ** 0.5
  test_clean = np.round(test_clean.reshape((m, n)))

  # img = Image.fromarray(test_noised[0].reshape((28,28)))
  # img.show()

  # output
  output_matrix = []
  out_f = open('final_result.csv', 'w')
  out_f.write('Id,Val\n')
  for i in xrange(100):
    for j in xrange(784):
      out_f.write('%d_%d,%d\n' % (i, j, test_clean[i,j]))
  out_f.close()


if __name__ == '__main__':
  main()
