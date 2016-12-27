from load_data import *
from sklearn.multiclass import OutputCodeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing

def main():
  print "Fetching Data..."
  train_data, train_labels, test_data = load_data()
  train_data_std = preprocessing.scale(train_data)
  test_data_std = preprocessing.scale(test_data)

  print "Start Training..."
  svc = svm.SVC(decision_function_shape='ovo')
  trained_model = svc.fit(train_data_std, train_labels)
  # print "Predicting Train Data..."
  # predictions = trained_model.predict(train_data_std)

  # print "Calculating Train Accuracy..."
  # correctness = 0.0
  # for c in (train_labels - predictions):
  #   if c == 0:
  #     correctness += 1.0

  # print float(correctness) / float(train_labels.size)

  print "Predicting Test Data..."
  predictions = trained_model.predict(test_data_std)
  print "Outputing..."
  output_matrix = []
  for (i, t) in zip(range(1, len(predictions)+1), predictions):
    output_matrix.append([i, t])
  output_matrix = np.mat(output_matrix)
  np.savetxt('final_result.csv', output_matrix, fmt=('%i', '%i'), delimiter=',',\
            header='id,category', comments='')


if __name__ == "__main__":
  main()