from itertools import combinations
import scipy.cluster.hierarchy as hcluster
import numpy
import matplotlib.pyplot as plt

def get_structure(boxes, lines):
  # rows = cluster_boxes(boxes, 1)
  # cols = cluster_boxes(boxes, 0)

  rate_combinations(boxes, lines)
  return ([], [])

def rate_combinations(boxes, lines):
  overall_row_scores = {}
  row_score_matrix = [[1.0 for x in range(len(boxes))] for y in range(len(boxes))]
  overall_col_scores = {}
  col_score_matrix = [[1.0 for x in range(len(boxes))] for y in range(len(boxes))]
  horiz_lines = lines[0]
  vert_lines = lines[1]

  for comb in combinations(enumerate(boxes), 2):
    row_scores = {}
    col_scores = {}

    i = comb[0][0]
    j = comb[1][0]

    box_1 = {
      'left': comb[0][1][0],
      'right': comb[0][1][0] + comb[0][1][2],
      'top': comb[0][1][1],
      'bottom': comb[0][1][1] + comb[0][1][3]
    }

    box_2 = {
      'left': comb[1][1][0],
      'right': comb[1][1][0] + comb[1][1][2],
      'top': comb[1][1][1],
      'bottom': comb[1][1][1] + comb[1][1][3]
    }

    # 1.) Their vertical (horizontal) centers align
    # May want to cut the factor down to 1.0 to make it a max of 1.0
    row_scores['center_align'] = 2.0 / (1.0 + abs(box_1['top'] + box_1['bottom'] - box_2['top'] - box_2['bottom']))
    col_scores['center_align'] = 2.0 / (1.0 + abs(box_1['left'] + box_1['right'] - box_2['left'] - box_2['right']))

    # 2.) Their left (top) edges align
    row_scores['top_align'] = 1.0 / (1.0 + abs(box_1['top'] - box_2['top']))
    col_scores['left_align'] = 1.0 / (1.0 + abs(box_1['left'] - box_2['left']))

    # 3.) Their right (bottom) edges align
    row_scores['bottom_align'] = 1.0 / (1.0 + abs(box_1['bottom'] - box_2['bottom']))
    col_scores['right_align'] = 1.0 / (1.0 + abs(box_1['right'] - box_2['right']))

    # 4.) If there is a line close to their left (above them)
    row_scores['top_line'] = calculate_preceding_line_score(box_1['top'], box_2['top'], horiz_lines)
    col_scores['left_line'] = calculate_preceding_line_score(box_1['left'], box_2['left'], vert_lines)

    # 5.) If there is a line close to their right (below them)
    row_scores['bottom_line'] = calculate_succeeding_line_score(box_1['bottom'], box_2['bottom'], horiz_lines)
    col_scores['right_line'] = calculate_succeeding_line_score(box_1['right'], box_2['right'], vert_lines)

    # 6.) They overlap significantly in their horizontal (vertical) range
    row_scores['vert_overlap'] = calculate_vertical_overlap(box_1, box_2)
    col_scores['horiz_overlap'] = calculate_horizontal_overlap(box_1, box_2)

    # 7.) I would like to add in a term regarding a shared strong score with a third object

    row_score = calculate_row_score(row_scores)
    col_score = calculate_col_score(col_scores)
    overall_row_scores[str(comb)] = row_score
    overall_col_scores[str(comb)] = col_score

    row_score_matrix[comb[0][0]][comb[1][0]] = row_score
    row_score_matrix[comb[1][0]][comb[0][0]] = row_score
    col_score_matrix[comb[0][0]][comb[1][0]] = col_score
    col_score_matrix[comb[1][0]][comb[0][0]] = col_score

  for comb in overall_row_scores:
    print('comb: ' + str(comb))
    print('row score: ' + str(overall_row_scores[comb]))
    print('col score: ' + str(overall_col_scores[comb]))

  row_clusters = cluster_structure(row_score_matrix)
  col_clusters = cluster_structure(col_score_matrix)

  print('Row clusters found:')
  for cluster in row_clusters:
    print('*****************')
    for i in cluster:
      print(boxes[i])

  print('------')
  print('Col clusters found:')
  for cluster in col_clusters:
    print('*****************')
    for i in cluster:
      print(boxes[i])

  print('Cluster size estimate: ' + str(len(row_clusters)) + 'x' + str(len(col_clusters)))

  print('done clustering')

def calculate_preceding_line_score(edge1, edge2, lines):
  min_edge = min(edge1, edge2)
  min_dist = float('inf')
  min_line = 0

  for line in lines:
    if line <= min_edge and min_edge - line < min_dist:
      min_dist = min_edge - line
      min_line = line

  # Could also just consider only the minimum, instead of both, but
  # I think both is probably better, for example in the case of two
  # With one far away from the other
  return 1.0 / (1.0 + edge1 - line + edge2 - line)

def calculate_succeeding_line_score(edge1, edge2, lines):
  max_edge = max(edge1, edge2)
  min_dist = float('inf')
  min_line = 0

  for line in lines:
    if line >= max_edge and max_edge - line < min_dist:
      min_dist = max_edge - line
      min_line = line

  # Could also just consider only the maximum, instead of both, but
  # I think both is probably better, for example in the case of two
  # With one far away from the other
  return 1.0 / (1.0 + line - edge1 + line - edge2)

def cluster_structure(score_matrix):
  clusters = []
  score_threshold = 1.3
  for i in range(len(score_matrix)):
    curr_cluster = set([i])
    for j in range(len(score_matrix)):
      curr_score = score_matrix[i][j]
      if curr_score > score_threshold:
        curr_cluster.add(j)

      clusters.append(curr_cluster)

  # Now we need to merge any clusters with shared elements
  # Based roughly on the algorithm from Niklas at:
  # http://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
  have_merged = True
  while have_merged:
    have_merged = False
    new_clusters = []
    while len(clusters) > 0:
      first = clusters[0]
      remaining = clusters[1:]
      clusters = []

      for cluster in remaining:
        if cluster.isdisjoint(first):
          clusters.append(cluster)
        else:
          have_merged = True
          first |= cluster

      new_clusters.append(first)

    clusters = new_clusters

  return clusters

# Returns percent overlap (length of overlap over min side interval length)
def calculate_vertical_overlap(box1, box2):
  inter_len = max(0, min(box1['bottom'], box2['bottom']) - max(box1['top'], box2['top']))
  min_size = min(box1['bottom'] - box1['top'], box2['bottom'] - box2['top'])

  return inter_len * 1.0 / min_size

# Returns percent overlap (length of overlap over min side interval length)
def calculate_horizontal_overlap(box1, box2):
  inter_len = max(0, min(box1['right'], box2['right']) - max(box1['left'], box2['left']))
  min_size = min(box1['right'] - box1['left'], box2['right'] - box2['left'])

  return inter_len * 1.0 / min_size

def calculate_row_score(scores):
  score = 0.0

  for score_type in scores:
    score += scores[score_type]

  return score

def calculate_col_score(scores):
  return calculate_row_score(scores) # Same for now

def cluster_boxes(boxes, offset):
  clusters = []

  for box in boxes:
    clustered = False
    box_edge_1 = box[offset]
    box_edge_2 = box[offset] + box[offset + 2]

    for i, cluster in enumerate(clusters):
      if clustered:
        break

      for c_box in cluster:
        c_box_edge_1 = c_box[offset]
        c_box_edge_2 = c_box[offset] + c_box[offset + 2]

        if c_box_edge_1 <= box_edge_1 <= c_box_edge_2 or c_box_edge_1 <= box_edge_2 <= c_box_edge_2:
          # We know this overlaps in the given direction
          # And want to add this to the cluster
          clusters[i].append(box)
          clustered = True
          break

    if not clustered:
      clusters.append([box]) # No existing matching cluster, so start a new one

  return clusters
