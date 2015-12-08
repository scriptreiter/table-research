predicted = {}

current_image = ''

def add_score(label, val):
  global predicted # Not necessarily necessary, but good to be explicit
  global current_image # ^^^

  if current_image not in predicted:
    predicted[current_image] = {}

  predicted[current_image][label] = val

def set_current_image(image):
  global current_image

  current_image = image

def evaluate():
  global predicted

  for image in predicted:
    print('Scores for image: ' + image)

    for score in predicted[image]:
      print('  ' + score + ': ' + str(predicted[image][score]))

    print()

  print('Evaluating the scores for this run of images')

  check_annotations()

def check_annotations():
  global predicted

  annotations = read_annotations()

  score_types = ['initial', 'combined', 'cluster', 'line_outside', 'line_inside']

  total_evaluated = 0
  total_correct = {}

  for score_type in score_types:
    total_correct[score_type + '_rows'] = 0
    total_correct[score_type + '_cols'] = 0
    total_correct[score_type + '_both'] = 0

    total_correct[score_type + '_rows_within_1'] = 0
    total_correct[score_type + '_cols_within_1'] = 0
    total_correct[score_type + '_both_within_1'] = 0

  for image in predicted:
    if image in annotations:
      total_evaluated += 1

      actual_rows = annotations[image]['rows']
      actual_cols = annotations[image]['cols']

      for score_type in score_types:
        guess_rows = predicted[image][score_type + '_rows']
        guess_cols = predicted[image][score_type + '_cols']

        if score_type == 'combined' and guess_rows != actual_rows:
          # import pdb;pdb.set_trace()
          print('Didn\'t get: ' + image)

        if guess_rows == actual_rows:
          total_correct[score_type + '_rows'] += 1

        if guess_cols == actual_cols:
          total_correct[score_type + '_cols'] += 1

        if guess_rows == actual_rows and guess_cols == actual_cols:
          total_correct[score_type + '_both']

        # Record if close
        if abs(guess_rows - actual_rows) < 2:
          total_correct[score_type + '_rows_within_1'] += 1

        if abs(guess_cols - actual_cols) < 2:
          total_correct[score_type + '_cols_within_1'] += 1

        if abs(guess_rows - actual_rows) < 2 and abs(guess_cols - actual_cols) < 2:
          total_correct[score_type + '_both_within_1']

    else:
      print('Image (' + image + ') not in annotations.')

  print('Evaluations:')
  for eval_type in total_correct:
    print('  Type: ' + eval_type)
    print('    ' + str(total_correct[eval_type]) + ' / ' + str(total_evaluated))
    print('    ' + str(total_correct[eval_type] * 100.0 / total_evaluated) + '%')
    print()

def read_annotations():
  annotations = {}

  with open('regents_guide.txt') as f:
    for line in f:
      parts = line.rstrip().split(',')

      img_name = parts[0]

      if len(parts) <= 3:
        annotations[img_name] = {
          'rows': int(parts[1]),
          'cols': int(parts[2])
        }

  return annotations
