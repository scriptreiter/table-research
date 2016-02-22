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

  import pdb;pdb.set_trace()
  check_annotations()

def check_annotations():
  global predicted

  annotations = read_annotations()

  # score_types = ['initial', 'cluster', 'line_outside', 'line_inside']
  score_types = ['cluster']

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

        if guess_rows == actual_rows:
          total_correct[score_type + '_rows'] += 1
        elif score_type == 'initial':
          print('Missed rows on ' + image + ' (actual=' + str(actual_rows) + ', guessed=' + str(guess_rows) + ')')

        if guess_cols == actual_cols:
          total_correct[score_type + '_cols'] += 1
        elif score_type == 'initial':
          print('Missed cols on ' + image + ' (actual=' + str(actual_cols) + ', guessed=' + str(guess_cols) + ')')

        if guess_rows == actual_rows and guess_cols == actual_cols:
          total_correct[score_type + '_both'] += 1

        # Record if close
        if abs(guess_rows - actual_rows) < 2:
          total_correct[score_type + '_rows_within_1'] += 1

        if abs(guess_cols - actual_cols) < 2:
          total_correct[score_type + '_cols_within_1'] += 1

        if abs(guess_rows - actual_rows) < 2 and abs(guess_cols - actual_cols) < 2:
          total_correct[score_type + '_both_within_1'] += 1

    else:
      print('Image (' + image + ') not in annotations.')

  print('Evaluations:')
  for eval_type in sorted(total_correct):
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

      if len(parts) <= 3 or int(parts[3]) == 1 or int(parts[3]) == 2:
        annotations[img_name] = {
          'rows': int(parts[1]),
          'cols': int(parts[2])
        }

  return annotations

def evaluate_cells(image, cells):
  threshold = 0.9
  gt_cells = []
  with open('ground_truth/alternate/' + image + '.txt') as f:
    for line in f:
      parts = line.split(',')
      gt_cells.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), ','.join(parts[4:]).rstrip('\n')))

  overlaps = [[] for x in range(len(gt_cells))]
  gt_overlaps = [[] for x in range(len(cells))]
  # Now, need to find correspondences
  for j, gt_cell in enumerate(gt_cells):
    overlaps.append([])
    for i, cell in enumerate(cells):
      overlap = get_overlap(cell, gt_cell)
      if overlap > threshold:
        overlaps[j].append((i, overlap))
        gt_overlaps[i].append((j, overlap))

  # This can result in a situation where a cell may be
  # marked for two difference gt cells. This can happen
  # if a merged cell overlaps two gt cells or more
  # We could do a more complex resolution, but for now
  # we'll do a basic check resolution of first one that
  # we encounter. Later, this could choose the best overlap or similar
  available = set(range(len(cells)))
  official = [None for x in range(len(gt_cells))]

  # If 1-1 relationship, we want to connect these
  for k in range(len(official)):
    if len(overlaps[k]) == 1 and len(gt_overlaps[overlaps[k][0][0]]) == 1:
      official[k] = overlaps[k][0][0]
      available.remove(official[k])

  # Now, go through the ones that have multiple
  # For now, try to pick the best overlap. Secondary
  # sorting by the index of the predicted box for now,
  # simply to know how it is sorted with equivalent overlap
  # As noted above, later we should do more complex things,
  # like overlap pixels, etc.

  # Go through once, and use any that only have single marked ones
  for k, idx in enumerate(official):
    if idx is None and len(overlaps[k]) > 0:
      overlaps[k].sort(key = lambda x: (x[1], x[0]), reverse = True)

      official[k] = next((x[1] for x in overlaps[k] if x[1] in available), None)
      # Using discard b/c if none of the indices are available,
      # we can avoid an extra check for the None that is returned
      available.discard(official[k])

  # Now we have a list of boxes matched to the ground truth boxes,
  # and just need to check the edit distances, and store the scores

def get_overlap(box_1, box_2):
  horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
  vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

  overlap_area = horiz_over * vert_over
  min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

  return overlap_area * 1.0 / min_area
