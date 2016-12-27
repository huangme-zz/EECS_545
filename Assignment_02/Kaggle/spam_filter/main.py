import pickle, os, csv
import numpy as np
from sklearn.naive_bayes import GaussianNB

with open(os.path.join('spam_filter_train.txt'), 'r') as f:
  targets = []
  for line in f:
    line = line.replace('\n', '').split('\t')[0]
    if line == 'ham':
      targets.append(0.0)
    if line == 'spam':
      targets.append(1.0)
  targets = np.array(targets)

with open(os.path.join('spam_filter_test.txt'), 'r') as f:
  orders = []
  for line in f:
    line = line.replace('\n', '').split('\t')[0]
    orders.append(line)

with open('trainFeatures.pkl', 'rb') as f:
  train_info = pickle.load(f)
with open('testFeatures.pkl', 'rb') as f:
  test_info = pickle.load(f) 

train_X = train_info.toarray()
test_X = test_info.toarray()
m, n = train_X.shape
train_t = np.zeros(m)

gnb = GaussianNB()
gnb.fit(train_X, targets)
pred_t = gnb.predict(test_X)
with open('final_result.csv', 'wb') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerow(['id', 'output'])
  for (number, result) in zip(orders, pred_t):
    csvwriter.writerow([number, result])