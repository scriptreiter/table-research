import clusterer
import os
import operator
import pickle
from itertools import combinations
from functools import cmp_to_key

def get_boxes(data, zoom_level, lines, feat_file, contour_boxes):
  raw_boxes = get_boxes_from_json(data, zoom_level)

  combined = combine_boxes(raw_boxes, lines, feat_file, contour_boxes)

  return combined, raw_boxes

def get_boxes_from_json(data, zoom_level = 1):
  boxes = []
  if 'regions' in data:
    for i, region in enumerate(data['regions']):
      if 'lines' in region:
        for j, line in enumerate(region['lines']):
          if 'words' in line:
            for k, word in enumerate(line['words']):
              if 'boundingBox' in word:
                bbox = [int(int(x) / zoom_level) for x in word['boundingBox'].split(',')]

                text = word['text'] if 'text' in word else ''

                bbox.append(text)
                bbox.append({'region': i, 'line': j, 'word': k})
                boxes.append(bbox)

  return boxes

def combine_boxes(boxes, lines, feat_file, contour_boxes):
  # To use the original merging, do this:
  # box_scores = score_boxes_orig(boxes, lines, feat_file)

  # print('scoring clusters for boxes')
  # score_clusters = clusterer.cluster_scores(box_scores, 4.0)

  # To use the classifier, do this:
  box_classifier_scores = score_boxes(boxes, lines, feat_file)

  box_scores = modify_box_scores(boxes, contour_boxes, box_classifier_scores)

  score_clusters = clusterer.cluster_scores(box_scores, 0.99)

  # Now need to translate score_cluster indices into boxes,
  # And then combine the boxes in each cluster

  clustered = [[boxes[idx] for idx in cluster] for cluster in score_clusters]

  combined = combine_clustered_boxes(clustered)

  return combined

  # return boxes # Soon return combined

# This method lowers scores for boxes to prevent merging across
# cell boundaries
def modify_box_scores(boxes, c_boxes, scores):
  threshold = 0.9
  # First calculate the cells which each box overlaps
  # This should be only one, but we're using a set to be
  # careful for now, in case thresholds are lowered
  overlaps = [set() for i in range(len(boxes))]
  for i, box_1 in enumerate(boxes):
    for j, box_2 in enumerate(c_boxes):
      horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
      vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

      overlap_area = horiz_over * vert_over
      min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

      if overlap_area * 1.0 / min_area > threshold:
        overlaps[i].add(j)

  # Now, for each combination of boxes, we want
  # to penalize if they do not overlap a shared cell
  # (if their overlap sets are disjoint)

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    if overlaps[i] != overlaps[j] and overlaps[i].isdisjoint(overlaps[j]):
      # This means they overlap different boxes
      # For now this will be an absolute penalty
      scores[i][j] = 0.0
      scores[j][i] = 0.0

  return scores

def combine_clustered_boxes(clusters):
  new_boxes = []

  for cluster in clusters:
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    labels = []

    # Sort the boxes in the cluster by y and then x
    # to preserve label ordering
    cluster.sort(key = lambda x: (x[1], x[0]))

    for box in cluster:
      min_x = min(min_x, box[0])
      min_y = min(min_y, box[1])
      max_x = max(max_x, box[0] + box[2])
      max_y = max(max_y, box[1] + box[3])
      labels.append(box[4])

    # We keep it x, y, width, height, labels for now
    # Although we later change it to be x1, y1, x2, y2, labels
    # For now we want to keep this paradigm
    new_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y, labels))

  return new_boxes

def score_boxes_orig(boxes, lines, feature_file):
  box_scores = [[0.0 for box in boxes] for box in boxes]

  # Remove since we're appending for now
  try:
    os.remove(feature_file)
  except OSError:
    pass

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    scores = {}
    features = {}

    # 1. Higher score the closer together they are
    # horiz and vert (percentage and flat)
    horiz_dist = max(0, max(box_1[0], box_2[0]) - min(box_1[0] + box_1[2], box_2[0] + box_2[2]))
    vert_dist = max(0, max(box_1[1], box_2[1]) - min(box_1[1] + box_1[3], box_2[1] + box_2[3]))

    min_horiz_range = min(box_1[2], box_2[2])
    min_vert_range = min(box_1[3], box_2[3])

    # scores['horiz_dist_pix'] = 1.0 / (1.0 + horiz_dist)
    # scores['horiz_dist_perc'] = 1.0 / (1.0 + (horiz_dist * 1.0 / min_horiz_range))

    # scores['vert_dist_pix'] = 1.0 / (1.0 + vert_dist)
    # scores['vert_dist_perc'] = 1.0 / (1.0 + (vert_dist * 1.0 / min_vert_range))

    dist = (horiz_dist ** 2 + vert_dist ** 2) ** 0.5
    min_range = (min_horiz_range ** 2 + min_vert_range ** 2) ** 0.5

    scores['dist_pix'] = 1.0 / (1.0 + dist)
    scores['dist_perc'] = 1.0 / (1.0 + (dist * 1.0 / min_range))
    features['dist_pix'] = dist
    features['dist_perc'] = dist * 1.0 / min_range

    # 2. Higher score if they overlap (horiz and vert)
    # both percentage and flat
    horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
    vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

    # scores['horiz_over_pix'] = horiz_over # Probably will need a weight < 1?
    # scores['horiz_over_perc'] = horiz_over * 1.0 / min_horiz_range

    # scores['vert_over_pix'] = vert_over
    # scores['vert_over_perc'] = vert_over * 1.0 / min_vert_range

    scores['overlap_pix'] = 1.0 * horiz_over * vert_over
    scores['overlap_perc'] = 1.0 * horiz_over * vert_over / (min_horiz_range * min_vert_range)
    features['overlap_pix'] = horiz_over * vert_over
    features['overlap_perc'] = scores['overlap_perc']

    # 3. Higher score if they are in an oxford api line together

    scores['share_line'] = 1.0 if box_1[5]['line'] == box_2[5]['line'] else 0.0
    features['share_line'] = 1 if box_1[5]['line'] == box_2[5]['line'] else 0

    # 4. Higher score if they are in an oxford api region together

    scores['share_region'] = 1.0 if box_1[5]['region'] == box_2[5]['region'] else 0.0
    features['share_region'] = 1 if box_1[5]['region'] == box_2[5]['region'] else 0

    # 5. Higher score if they're aligned horizontally ?

    # 6. Higher score if they're aligned vertically ?

    # 7. Lower score if they have a line between them vertically or horizontally

    # Check for horizontal line b/w boxes
    scores['no_div_horiz_line'] = 0.0 if line_between(box_1, box_2, lines, 1) else 1.0
    features['no_div_horiz_line'] = 0 if line_between(box_1, box_2, lines, 1) else 1

    # Check for vertical line b/w boxes
    scores['no_div_vert_line'] = 0.0 if line_between(box_1, box_2, lines, 0) else 1.0
    features['no_div_vert_line'] = 0 if line_between(box_1, box_2, lines, 0) else 1

    # Additional features?
    # Ratio of areas
    # Area of box 1
    # Area of box 2
    # Left to right distance, and right to left distance
    # and top to bottom and bottom to top

    score = combine_scores(scores)
    record_features(box_1, box_2, features, feature_file)

    box_scores[i][j] = score
    box_scores[j][i] = score

  print('Features:')
  for x in sorted(features.keys()):
    print(x)

  return box_scores

def score_boxes(boxes, lines, feature_file):
  box_scores = [[0.0 for box in boxes] for box in boxes]

  # Remove since we're appending for now
  try:
    os.remove(feature_file)
  except OSError:
    pass

  with open('regents/test_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

  features = {}

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    scores = {}
    features = {}

    # 1. Higher score the closer together they are
    # horiz and vert (percentage and flat)
    horiz_dist = max(0, max(box_1[0], box_2[0]) - min(box_1[0] + box_1[2], box_2[0] + box_2[2]))
    vert_dist = max(0, max(box_1[1], box_2[1]) - min(box_1[1] + box_1[3], box_2[1] + box_2[3]))

    min_horiz_range = min(box_1[2], box_2[2])
    min_vert_range = min(box_1[3], box_2[3])

    # scores['horiz_dist_pix'] = 1.0 / (1.0 + horiz_dist)
    # scores['horiz_dist_perc'] = 1.0 / (1.0 + (horiz_dist * 1.0 / min_horiz_range))

    # scores['vert_dist_pix'] = 1.0 / (1.0 + vert_dist)
    # scores['vert_dist_perc'] = 1.0 / (1.0 + (vert_dist * 1.0 / min_vert_range))

    dist = (horiz_dist ** 2 + vert_dist ** 2) ** 0.5
    min_range = (min_horiz_range ** 2 + min_vert_range ** 2) ** 0.5

    scores['dist_pix'] = 1.0 / (1.0 + dist)
    scores['dist_perc'] = 1.0 / (1.0 + (dist * 1.0 / min_range))
    features['dist_pix'] = dist
    features['dist_perc'] = dist * 1.0 / min_range

    # 2. Higher score if they overlap (horiz and vert)
    # both percentage and flat
    horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
    vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

    # scores['horiz_over_pix'] = horiz_over # Probably will need a weight < 1?
    # scores['horiz_over_perc'] = horiz_over * 1.0 / min_horiz_range

    # scores['vert_over_pix'] = vert_over
    # scores['vert_over_perc'] = vert_over * 1.0 / min_vert_range

    scores['overlap_pix'] = 1.0 * horiz_over * vert_over
    scores['overlap_perc'] = 1.0 * horiz_over * vert_over / (min_horiz_range * min_vert_range)
    features['overlap_pix'] = horiz_over * vert_over
    features['overlap_perc'] = scores['overlap_perc']

    # 3. Higher score if they are in an oxford api line together

    scores['share_line'] = 1.0 if box_1[5]['line'] == box_2[5]['line'] else 0.0
    features['share_line'] = 1 if box_1[5]['line'] == box_2[5]['line'] else 0

    # 4. Higher score if they are in an oxford api region together

    scores['share_region'] = 1.0 if box_1[5]['region'] == box_2[5]['region'] else 0.0
    features['share_region'] = 1 if box_1[5]['region'] == box_2[5]['region'] else 0

    # 5. Higher score if they're aligned horizontally ?

    # 6. Higher score if they're aligned vertically ?

    # 7. Lower score if they have a line between them vertically or horizontally

    # Check for horizontal line b/w boxes
    scores['no_div_horiz_line'] = 0.0 if line_between(box_1, box_2, lines, 1) else 1.0
    features['no_div_horiz_line'] = 0 if line_between(box_1, box_2, lines, 1) else 1

    # Check for vertical line b/w boxes
    scores['no_div_vert_line'] = 0.0 if line_between(box_1, box_2, lines, 0) else 1.0
    features['no_div_vert_line'] = 0 if line_between(box_1, box_2, lines, 0) else 1

    # Additional features?
    # Ratio of areas
    box_1_area = box_1[2] * box_1[3]
    box_2_area = box_2[2] * box_2[3]
    features['ratio_of_areas_1'] = box_1_area * 1.0 / box_2_area
    features['ratio_of_areas_2'] = box_2_area * 1.0 / box_1_area
    # Area of box 1
    features['box_1_area'] = box_1_area
    # Area of box 2
    features['box_2_area'] = box_2_area
    # Left to right distance, and right to left distance
    features['ltr_dist'] = min(abs(box_1[0] - (box_2[0] + box_2[2])), box_2[0] - (box_1[0] + box_1[2]))
    # and top to bottom and bottom to top
    features['ttb_dist'] = min(abs(box_1[1] - (box_2[1] + box_2[3])), box_2[1] - (box_1[1] + box_1[3]))

    # Line in between ignoring overlap
    features['no_horiz_line_in_middle'] = 0 if line_in_middle(box_1, box_2, lines, 1) else 1
    features['no_vert_line_in_middle'] = 0 if line_in_middle(box_1, box_2, lines, 0) else 1

    # Distance between centers
    features['dist_bw_vert_centers'] = abs((box_1[0] + box_1[0] + box_1[2]) - (box_2[0] + box_2[0] + box_2[2])) / 2.0
    features['dist_bw_horiz_centers'] = abs((box_1[1] + box_1[1] + box_1[3]) - (box_2[1] + box_2[1] + box_2[3])) / 2.0

    # Distance between tops
    features['dist_bw_lefts'] = abs(box_1[0] - box_2[0])
    features['dist_bw_tops'] = abs(box_1[1] - box_2[1])

    # Distance between bottoms
    features['dist_bw_rights'] = abs((box_1[0] + box_1[2]) - (box_2[0] + box_2[2]))
    features['dist_bw_bottoms'] = abs((box_1[1] + box_1[3]) - (box_2[1] + box_2[3]))

    score = get_classifier_score(features, classifier)
    record_features(box_1, box_2, features, feature_file)

    box_scores[i][j] = score
    box_scores[j][i] = score

  if len(features) > 0:
    print('Features:')
    for x in sorted(features.keys()):
      print(x)

  return box_scores

def get_classifier_score(features, classifier):
  features = [features[x] for x in sorted(features.keys())]
  pred = classifier.predict([features])

  return pred[0] * 1.0

def record_features(box_1, box_2, features, feature_file):
  with open(feature_file, 'a') as f:
    f.write(','.join([str(x) for x in (box_1[0:4] + box_2[0:4])]))

    for x in sorted(features.keys()):
      f.write(',' + str(features[x]))

    f.write('\n')

# Line in between without intersection
def line_in_middle(box1, box2, lines, offset):
  for line in lines[(offset + 1) % 2]:
    # Should this be a comparison only b/w the innermost values?

    # Check if box1, line, box2
    is_bw_1 = box1[offset] + box1[offset + 2] <= line['border'] <= box2[offset]

    # Check if box2, line, box1
    is_bw_2 = box2[offset] + box2[offset + 2] <= line['border'] <= box1[offset]

    if is_bw_1 or is_bw_2:
      return True
  return False

def line_between(box1, box2, lines, offset):
  for line in lines[(offset + 1) % 2]:
    # Should this be a comparison only b/w the innermost values?

    # Check if box1, line, box2
    is_bw_1 = box1[offset] + box1[offset + 2] <= line['border'] <= box2[offset]

    # Check if box2, line, box1
    is_bw_2 = box2[offset] + box2[offset + 2] <= line['border'] <= box1[offset]

    alt_offset = (offset + 1) % 2

    # Check if the line overlaps with box1
    intersects_1 = max(0, min(line['end'], box1[alt_offset] + box1[alt_offset + 2]) - max(line['start'], box1[alt_offset]))

    # Check if the line overlaps with box2
    intersects_2 = max(0, min(line['end'], box2[alt_offset] + box2[alt_offset + 2]) - max(line['start'], box2[alt_offset]))
    if (is_bw_1 or is_bw_2) and (intersects_1 or intersects_2):
      return True

  return False

def combine_scores(scores):
  total = 0.0

  # Naive combination for now
  for score_type in scores:
    total += scores[score_type]

  return total

def add_labels(boxes, label_boxes, threshold):
  labeled = []
  for box in boxes:
    labels = []
    for label_box in label_boxes:
      horiz_over = max(0, min(box[0] + box[2], label_box[0] + label_box[2]) - max(box[0], label_box[0]))
      vert_over = max(0, min(box[1] + box[3], label_box[1] + label_box[3]) - max(box[1], label_box[1]))
      overlap_area = horiz_over * vert_over
      min_area = min(box[2] * box[3], label_box[2] * label_box[3])
      if overlap_area * 1.0 / min_area > threshold:
        labels.append(label_box)

    # Double-check this sorting order, but this is y, then x
    label = [x[4] for x in sorted(labels, key=cmp_to_key(label_comparator))]
    # import pdb;pdb.set_trace()

    labeled.append((box[0], box[1], box[2], box[3], label))

  return labeled

# Comparator for labels with some variance allowed for lines with variable toplines
def label_comparator(a, b):
  top_dist = abs(a[1] - b[1])
  min_height = min(a[3], b[3])

  if top_dist * 1.0 / min_height > 0.25:
    # This means the difference of the tops
    # is significant, so sort by it
    return a[1] - b[1]

  # Otherwise we want to sort by the left edges
  return a[0] - b[0]

# This method merges two groups of boxes by calculating overlap
# Overlapping boxes are connected
# Then any X-many relationships are resolved into the 'many' side
def merge_box_groups(group_1, group_2, threshold, bbox):
  conns_1 = [[] for i in range(len(group_1))]
  conns_2 = [[] for i in range(len(group_2))]

  for i, box_1 in enumerate(group_1):
    # Should these check off threshold, instead?
    for j, box_2 in enumerate(group_2):
      horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
      vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

      overlap_area = horiz_over * vert_over
      min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

      if overlap_area * 1.0 / min_area > threshold:
        conns_1[i].append(j)
        conns_2[j].append(i)

  # We have two ways we could approach this filtering:
  # 1. Find ones with 1-1 or 1-0, and add the largest of them
  # 2. For all of them, add the ones in a 1-many, add the many
  # and then filter with a set. I think this should be equivalent
  # given that there are no intragroup overlaps
  # and thus am going to do option 1 for now.

  # We only want boxes that overlap the bounding box, but
  # If it isn't a 1-0, we know the overlap with a cell box is
  # above the threshold, and don't need to check this case
  # Thus we only check 1-0 ones for now (for which we could get
  # away with only checking the second group, technically)

  merged = []
  for i, conns in enumerate(conns_1):
    if len(conns) == 0:
      if box_overlap(group_1[i], bbox) > threshold:
        merged.append(group_1[i])
    elif len(conns) == 1:
      selector = None
      if len(conns_2[conns[0]]) == 1:
        # This is a 1-1 correspondence, and we want the biggest
        # box, as it is likely more accurate
        selector = get_largest
      else:
        # This is a many-1 correspondence, and we want to take
        # the 1, being the smaller box
        # We may be able to make a correct assumption about
        # which is smallest, but we'll check for now
        selector = get_smallest
        
      merged.append(selector(group_1[i], group_2[conns[0]]))

  for i, conns in enumerate(conns_2):
    if len(conns) == 0:
      if box_overlap(group_2[i], bbox) > threshold:
        merged.append(group_2[i])
    # This needs to check if we already processed, to avoid dedups
    # We will have processed it only if it is a 1-1
    elif len(conns) == 1 and len(conns_1[conns[0]]) != 1:
      # This is guaranteed to be a many-1, thus we
      # can simply take the smallest
      merged.append(get_smallest(group_2[i], group_1[conns[0]]))

  return merged

# TODO: Refactor so that everything uses this
# Maybe move it to a more centralized location
def box_overlap(box_1, box_2):
  horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
  vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

  overlap_area = horiz_over * vert_over
  min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

  return overlap_area * 1.0 / min_area

# Return the larger of the two boxes
def get_largest(box_1, box_2):
  area_1 = box_1[2] * box_1[3]
  area_2 = box_2[2] * box_2[3]

  if area_1 > area_2:
    return box_1
  
  return box_2

def get_smallest(box_1, box_2):
  area_1 = box_1[2] * box_1[3]
  area_2 = box_2[2] * box_2[3]

  if area_1 <= area_2:
    return box_1
  
  return box_2
