import os
import pickle

from sklearn.ensemble import RandomForestClassifier

# Path to files listing box combos in the same order as the labels
key_path = 'regents/combos/keys/'

# Path to files labeling box combos as merge (1) or no merge (0)
label_path = 'regents/combos/labels/'

# Path to files keeping box combos and associated features
feature_path = 'regents/combos/features/'

# Path to save the classifier
classifier_path = 'regents/classifier.pkl'

# Path to save the list of files in the training set
train_set_path = 'regents/train_set.txt'

# Path to save the list of files in the testing set
test_set_path = 'regents/test_set.txt'

def setup():
  key_files = [key_file for key_file in os.listdir(key_path)]

  train_labels = []
  test_labels = []
  train_set = []

  train_features = []
  test_features = []
  test_set = []

  for key_file in key_files:
    l, f = get_image_info(key_file)

    if len(train_labels) < 35000:
      train_set.append(key_file)
      train_labels += l
      train_features += f
    else:
      test_set.append(key_file)
      test_labels += l
      test_features += f

  train_features = trim_features(train_features)
  test_features = trim_features(test_features)

  forest = RandomForestClassifier(n_estimators=100)
  forest.fit(train_features, train_labels)

  tn_count = 0
  tp_count = 0
  fn_count = 0
  fp_count = 0

  false_negs = []
  false_poss = []
  for (l, f) in zip(test_labels, test_features):
    pred = forest.predict([f])

    if l == 0:
      if pred[0] == 0:
        tn_count += 1
      else:
        fp_count += 1
        false_poss.append(f)
    else:
      if pred[0] == 1:
        tp_count += 1
      else:
        fn_count += 1
        false_negs.append(f)

  print('Accuracy: ' + str((tn_count + tp_count) * 1.0 / len(test_labels)))
  print('Recall: ' + str(tp_count * 1.0 / (tp_count + fn_count)))
  print('Precision: ' + str(tp_count * 1.0 / (tp_count + fp_count)))

  with open(classifier_path, 'wb') as f:
    pickle.dump(forest, f)

  with open(train_set_path, 'w') as f:
    f.write(' \n'.join(train_set))

  with open(test_set_path, 'w') as f:
    f.write(' \n'.join(test_set))

  import pdb;pdb.set_trace()
  print('Done')

def trim_features(feats):
  return [x[:-3] + x[-1:] for x in feats]

def get_image_info(key_file):
  with open(label_path + key_file, 'r') as f:
    labels = [int(x.strip('\n')) for x in f]

  with open(feature_path + key_file, 'r') as f:
    features = [[float(y) for y in x.split(',')[8:]] for x in f]

  return (labels, features)

if __name__ == '__main__':
  setup()
