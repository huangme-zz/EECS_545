import math
import numpy as np

def readData(filename):
  in_f = open(filename, 'rb')

  feature_map = []
  categories = []
  for line in in_f:
    data = line.strip().split(',')
    features = [float(entry) for entry in data[:-1]]
    category = float(data[-1])
    feature_map.append(features)
    categories.append(category)

  in_f.close()
  return feature_map, categories

def getMedian(l):
  l = sorted(l)
  n = len(l)
  if n % 2:
    return l[n/2]
  else:
    return float(l[n/2 - 1] + l[n/2]) / 2.0

def preprocess(l):
  med = getMedian(l)
  result = []
  for i in xrange(len(l)):
    if l[i] > med:
      result.append(1.0)
    else:
      result.append(0.0)
  return result

def main():
  # read data from files
  train_feature_map, train_categories = readData('spambase.train')
  test_feature_map, test_categories = readData('spambase.test')

  # preprocess
  total_feature_map = train_feature_map + test_feature_map
  for j in xrange(len(total_feature_map[0])):
    l = [total_feature_map[i][j] for i in range(len(total_feature_map))]
    l = preprocess(l)
    for i in xrange(len(total_feature_map)):
      total_feature_map[i][j] = l[i]
  train_feature_map = total_feature_map[:2000]
  test_feature_map = total_feature_map[2000:]

  # training step
  N_spam = 0.0
  N_nonspam = 0.0
  for elt in train_categories:
    if elt == 1:
      N_spam += 1.0
    else:
      N_nonspam += 1.0
  phi_spam = math.log(N_spam / (N_spam + N_nonspam))
  phi_nonspam = math.log(N_nonspam / (N_spam + N_nonspam))


  N_feature_spam_list = []
  N_feature_nonspam_list = []
  for j in xrange(len(train_feature_map[0])):
    N_feature_spam = 0.0
    N_feature_nonspam = 0.0
    for i in xrange(len(train_feature_map)):
      if train_feature_map[i][j] == 1:
        if train_categories[i] == 1:
          N_feature_spam += 1.0
        else:
          N_feature_nonspam += 1.0
    N_feature_spam_list.append(N_feature_spam)
    N_feature_nonspam_list.append(N_feature_nonspam)
  S_spam = sum(N_feature_spam_list)
  S_nonspam = sum(N_feature_nonspam_list)

  mu_spam_list = [math.log(N/S_spam) for N in N_feature_spam_list]
  mu_nonspam_list = [math.log(N/S_nonspam) for N in N_feature_nonspam_list]

  # prediction part
  predictions = []
  for i in xrange(len(test_feature_map)):
    result_spam = phi_spam
    result_nonspam = phi_nonspam
    for j in xrange(len(test_feature_map[0])):
      if test_feature_map[i][j] == 1:
        result_spam += mu_spam_list[j]
        result_nonspam += mu_nonspam_list[j]

    # predict
    if result_nonspam < result_spam:
      predictions.append(1.0)
    else:
      predictions.append(0.0)

  # calculate error
  error = 0.0
  for (y, t) in zip(predictions, test_categories):
    if y != t:
      error += 1.0
  print error / float(len(test_categories))

  # sanity check
  error = 0.0
  for y in predictions:
    if y == 1:
      error += 1.0

  print error / float(len(test_categories))









if __name__ == '__main__':
  main()