import clusterer
from itertools import combinations

def get_boxes(data, zoom_level, lines):
  raw_boxes = get_boxes_from_json(data, zoom_level)

  combined = combine_boxes(raw_boxes, lines)

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

def combine_boxes(boxes, lines):
  box_scores = score_boxes(boxes, lines)

  # print('scoring clusters for boxes')
  score_clusters = clusterer.cluster_scores(box_scores, 4.0)

  # Now need to translate score_cluster indices into boxes,
  # And then combine the boxes in each cluster

  clustered = [[boxes[idx] for idx in cluster] for cluster in score_clusters]

  combined = combine_clustered_boxes(clustered)

  return combined

  # return boxes # Soon return combined

def combine_clustered_boxes(clusters):
  new_boxes = []

  for cluster in clusters:
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    labels = []

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

def score_boxes(boxes, lines):
  box_scores = [[0.0 for box in boxes] for box in boxes]

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    scores = {}

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

    # 3. Higher score if they are in an oxford api line together

    scores['share_line'] = 1.0 if box_1[5]['line'] == box_2[5]['line'] else 0.0

    # 4. Higher score if they are in an oxford api region together

    scores['share_region'] = 1.0 if box_1[5]['region'] == box_2[5]['region'] else 0.0

    # 5. Higher score if they're aligned horizontally ?

    # 6. Higher score if they're aligned vertically ?

    # 7. Lower score if they have a line between them vertically or horizontally

    # Check for horizontal line b/w boxes
    scores['no_div_horiz_line'] = 0.0 if line_between(box_1, box_2, lines, 1) else 1.0

    # Check for vertical line b/w boxes
    scores['no_div_vert_line'] = 0.0 if line_between(box_1, box_2, lines, 0) else 1.0

    score = combine_scores(scores)

    box_scores[i][j] = score
    box_scores[j][i] = score

  return box_scores

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
