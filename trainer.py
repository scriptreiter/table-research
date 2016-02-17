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

def setup():
  key_files = [key_file for key_file in os.listdir(key_path)]

  labels = []

  features = []

  for key_file in key_files:
    l, f = get_image_info(key_file)
    labels += l
    features += f

  num_train = int(len(labels) * 3 / 4)

  train_labels = labels[:num_train]
  train_features = features[:num_train]

  test_labels = labels[num_train:]
  test_features = features[num_train:]

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

  import pdb;pdb.set_trace()
  print('Done')

def get_image_info(key_file):
  with open(label_path + key_file, 'r') as f:
    labels = [int(x.strip('\n')) for x in f]

  with open(feature_path + key_file, 'r') as f:
    features = [[float(y) for y in x.split(',')[8:]] for x in f]

  return (labels, features)

if __name__ == '__main__':
  setup()
