import numpy as np
from numpy.linalg import pinv
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from sklearn import svm
import math, csv

NumIteration = 1000
ita = 0.01
C = 3
a = lambda j : float(ita) / float(1.0 + j * ita)

def readFile(filename):
  data = np.genfromtxt('./%s' % filename, delimiter=',')
  return data

def getAccuracy(w, b, data, labels):
  data = data.T
  M, N = data.shape

  correctness = 0
  for i in xrange(0, N):
    x = data[:, i]
    t = labels[i]

    if (np.dot(w.T, x) + b) > 0:
      if t == 1:
        correctness += 1
    else:
      if t == -1:
        correctness += 1

  return float(correctness) / float(N)

def show_image(image, fig, show=False):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    if show:
      plt.show()










# Batch Gradient Descent
def trainBatchDescent(data, labels):
  data = data.T
  M, N = data.shape
  w = np.zeros(M)
  b = 0.0
  acc_list = []
  for j in xrange(NumIteration):
    w_grad = w
    b_grad = 0.0
    for i in xrange(N):
      x = data[:, i]
      t = labels[i]

      if (t * (np.dot(w.T, x) + b)) < 1:
        w_grad = w_grad - C * t * x
        b_grad = b_grad - C * t

    w = w - a(j+1) * w_grad
    b = b - a(j+1) * b_grad

    correctness = 0
    for i in xrange(0, N):
      x = data[:, i]
      t = labels[i]

      if (np.dot(w.T, x) + b) > 0:
        if t == 1:
          correctness += 1
      else:
        if t == -1:
          correctness += 1
    acc_list.append(float(correctness) / float(N))


  return w, b, acc_list

def Batch():
  print "Doing Batch Descent..."
  training_data = readFile('digits_training_data.csv')
  training_labels = readFile('digits_training_labels.csv')
  test_data = readFile('digits_test_data.csv')
  test_labels = readFile('digits_test_labels.csv')

  training_data_std = preprocessing.scale(training_data)
  test_data_std = preprocessing.scale(test_data)

  for i in xrange(0, training_labels.shape[0]):
    if training_labels[i] == 4:
      training_labels[i] = -1
    if training_labels[i] == 9:
      training_labels[i] = 1

  for i in xrange(0, test_labels.shape[0]):
    if test_labels[i] == 4:
      test_labels[i] = -1
    if test_labels[i] == 9:
      test_labels[i] = 1

  w, b, acc_list = trainBatchDescent(training_data, training_labels)
  acc = getAccuracy(w, b, training_data, training_labels)
  print "training accuracy: %f" % acc
  print "Finished Batch Descent"
  return acc_list













# Stochastic Gradient Descent
def trainStochasticDescent(data, labels):
  data = data.T
  M, N = data.shape
  w = np.zeros(M)
  b = 0
  acc_list = []

  for j in xrange(NumIteration):
    for i in np.random.permutation(N):
      x = data[:, i]
      t = labels[i]

      w_grad = 1.0 / float(N) * w
      b_grad = 0.0
      if (t * (np.dot(w.T, x) + b)) < 1:
        w_grad -= C * t * x
        b_grad -= C * t

      w = w - a(j+1) * w_grad
      b = b - a(j+1) * b_grad

    correctness = 0
    for i in xrange(0, N):
      x = data[:, i]
      t = labels[i]

      if (np.dot(w.T, x) + b) > 0:
        if t == 1:
          correctness += 1
      else:
        if t == -1:
          correctness += 1
    acc_list.append(float(correctness) / float(N))

  return w, b, acc_list

def Stochastic():
  print "Doing Stochastic Descent..."
  training_data = readFile('digits_training_data.csv')
  training_labels = readFile('digits_training_labels.csv')
  test_data = readFile('digits_test_data.csv')
  test_labels = readFile('digits_test_labels.csv')

  training_data_std = preprocessing.scale(training_data)
  test_data_std = preprocessing.scale(test_data)

  for i in xrange(0, training_labels.shape[0]):
    if training_labels[i] == 4:
      training_labels[i] = -1
    if training_labels[i] == 9:
      training_labels[i] = 1

  for i in xrange(0, test_labels.shape[0]):
    if test_labels[i] == 4:
      test_labels[i] = -1
    if test_labels[i] == 9:
      test_labels[i] = 1

  w, b, acc_list = trainStochasticDescent(training_data, training_labels)
  acc = getAccuracy(w, b, training_data, training_labels)
  print "training accuracy: %f" % acc
  print "Finished Stochastic Descent"
  return acc_list








# RBF Kernel for SVM
def SVM_RBF_Kernel():
  print "Doing RBF Kernel..."
  training_data = readFile('digits_training_data.csv')
  training_labels = readFile('digits_training_labels.csv')
  test_data = readFile('digits_test_data.csv')
  test_labels = readFile('digits_test_labels.csv')
  for i in xrange(0, training_labels.shape[0]):
    if training_labels[i] == 4:
      training_labels[i] = -1
    if training_labels[i] == 9:
      training_labels[i] = 1

  for i in xrange(0, test_labels.shape[0]):
    if test_labels[i] == 4:
      test_labels[i] = -1
    if test_labels[i] == 9:
      test_labels[i] = 1

  training_data_std = preprocessing.scale(training_data)
  test_data_std = preprocessing.scale(test_data)

  img_list = []
  
  rbf_svc = svm.SVC(kernel='rbf', C=1)
  rbf_svc.fit(training_data_std, training_labels)

  predictions = rbf_svc.predict(training_data_std)
  correctness = sum(((predictions + training_labels) / 2) ** 2)
  acc = float(correctness) / float(training_labels.size)
  print "training accuracy: %f" % acc

  predictions = rbf_svc.predict(test_data_std)
  correctness = sum(((predictions + test_labels) / 2) ** 2)
  acc = float(correctness) / float(test_labels.size)
  print "test accuracy: %f" % acc

  imgHeight = np.sqrt(test_data.shape[1])
  n = 0
  for i in xrange(predictions.size):
    if predictions[i] != test_labels[i]:
      if predictions[i] == -1:
        img_list.append((test_data[i].reshape((imgHeight, imgHeight)), 4))
      else:
        img_list.append((test_data[i].reshape((imgHeight, imgHeight)), 9))
      n += 1
    if n == 5:
      break


  print "Finished RBF Kernel"
  return img_list








# Linear Discriminant Analysis (LDA)
def trainLDA(data, labels):
  N4 = 0.0
  x4 = np.zeros(data.shape[1])
  N9 = 0.0
  x9 = np.zeros(data.shape[1])
  for i in xrange(labels.size):
    x = data[i, :]
    t = labels[i]

    if t == 1:
      N9 += 1.0
      x9 = x9 + x

    else:
      N4 += 1.0
      x4 = x4 + x

  mu4 = np.matrix(x4 / N4).T
  mu9 = np.matrix(x9 / N9).T

  sigma = np.matrix(np.zeros((data.shape[1], data.shape[1])))
  for i in xrange(labels.size):
    x = np.matrix(data[i, :]).T
    t = labels[i]

    if t == -1:
      sigma = sigma + (x - mu4) * (x - mu4).T
    else:
      sigma = sigma + (x - mu9) * (x - mu9).T
  sigma = sigma / (N4 + N9)

  return N9 / (N4 + N9), mu4, mu9, sigma



def predictLDA(test_data, model):
  phi9, mu4, mu9, sigma = model
  sigma_inv = pinv(sigma)

  predictions = []
  for i in xrange(test_data.shape[0]):
    x = np.matrix(test_data[i, :]).T
    P4 = -1 / 2 * (x - mu4).T * sigma_inv * (x - mu4) + np.log(1.0 - phi9) 
    P9 = -1 / 2 * (x - mu9).T * sigma_inv * (x - mu9) + np.log(phi9)
    if P4 < P9:
      predictions.append(1)
    else:
      predictions.append(-1)

  return np.array(predictions)



def LDA():
  print "Doing LDA..."
  training_data = readFile('digits_training_data.csv')
  training_labels = readFile('digits_training_labels.csv')
  test_data = readFile('digits_test_data.csv')
  test_labels = readFile('digits_test_labels.csv')
  for i in xrange(0, training_labels.shape[0]):
    if training_labels[i] == 4:
      training_labels[i] = -1
    if training_labels[i] == 9:
      training_labels[i] = 1

  for i in xrange(0, test_labels.shape[0]):
    if test_labels[i] == 4:
      test_labels[i] = -1
    if test_labels[i] == 9:
      test_labels[i] = 1

  training_data_std = preprocessing.scale(training_data)
  test_data_std = preprocessing.scale(test_data)

  img_list = []
  
  model = trainLDA(training_data_std, training_labels)

  predictions = predictLDA(training_data_std, model)
  correctness = sum(((predictions + training_labels) / 2) ** 2)
  acc = float(correctness) / float(training_labels.size)
  print "training accuracy: %f" % acc

  predictions = predictLDA(test_data_std, model)
  correctness = sum(((predictions + test_labels) / 2) ** 2)
  acc = float(correctness) / float(test_labels.size)
  print "test accuracy: %f" % acc

  imgHeight = np.sqrt(test_data.shape[1])
  n = 0
  for i in xrange(predictions.size):
    if predictions[i] != test_labels[i]:
      if predictions[i] == -1:
        img_list.append((test_data[i].reshape((imgHeight, imgHeight)), 4))
      else:
        img_list.append((test_data[i].reshape((imgHeight, imgHeight)), 9))
      n += 1
    if n == 5:
      break


  print "Finished LDA"
  return img_list




def main():
  batch_acc_list = Batch()
  stoch_acc_list = Stochastic()
  
  fig = plt.figure()
  plt.plot(np.log(range(NumIteration)[1:]), batch_acc_list[1:], \
           np.log(range(NumIteration)[1:]), stoch_acc_list[1:])
  plt.xlabel('Log Iteration')
  plt.ylabel('Training Accuracy')
  fig.savefig('1.png')

  img_list = SVM_RBF_Kernel()
  n = 1
  for img, pred in img_list:
    fig = plt.figure()
    show_image(img, fig)
    fig.savefig('mis_img_%d_label_%d.png' % (n, pred))
    n += 1

  img_list = LDA()
  n = 6
  for img, pred in img_list:
    fig = plt.figure()
    show_image(img, fig)
    fig.savefig('mis_img_%d_label_%d.png' % (n, pred))
    n += 1

if __name__ == '__main__':
  main()