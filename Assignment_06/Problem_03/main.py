from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import cPickle

def getData(filename, flag=True):
  in_f = open(filename, "r")
  loaded_obj = cPickle.load(in_f)
  in_f.close()
  X = loaded_obj["data"]
  if flag:
    t = loaded_obj["labels"]
    return X, t
  return X

def main():
  X_train, t_train = getData("train.pkl")
  X_test = getData("test.pkl", False)

  print "start"
  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(X_train, t_train)
  t_pred = clf.predict(X_test)
  print "end"
  
  # # Compute PCA
  # n_components=150
  # h = 32
  # w = 96
  # t0 = time()
  # print "start RandomizedPCA"
  # pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
  # print "done in %0.3fs" % (time() - t0)

  # eigenfaces = pca.components_.reshape((n_components, h, w))

  # t0 = time()
  # X_train_pca = pca.transform(X_train)
  # X_test_pca = pca.transform(X_test)
  # print "done in %0.3fs" % (time() - t0)

  # print "Fitting the classifier to the training set"
  # t0 = time()
  # param_grid = {'C': [5e3, 1e4],
  #               'gamma': [0.0005, 0.001]}
  # clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
  # clf = clf.fit(X_train_pca, t_train)
  # print "done in %0.3fs" % (time() - t0)
  # print "best estimator found by grid search:"
  # print clf.best_estimator_

  # print("Predicting people's names on the test set")
  # t0 = time()
  # t_pred = clf.predict(X_test_pca)
  # print("done in %0.3fs" % (time() - t0))

  # output
  output_matrix = []
  out_f = open('final_result.csv', 'w')
  out_f.write('id,category\n')
  n = 1
  for c in t_pred:
    out_f.write('%d,%d\n' % (n,c))
    n += 1
  out_f.close()


main()