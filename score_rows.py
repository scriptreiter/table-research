from itertools import combinations

def get_structure(boxes, lines):
  # rows = cluster_boxes(boxes, 1)
  # cols = cluster_boxes(boxes, 0)

  rate_combinations(boxes)
  return ([], [])

def rate_combinations(boxes):
  overall_row_scores = {}
  overall_col_scores = {}

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
    row_scores['center_align'] = 2.0 / (1.0 + abs(box_1['top'] + box_1['bottom'] - box_2['top'] - box_2['bottom']))
    col_scores['center_align'] = 2.0 / (1.0 + abs(box_1['left'] + box_1['right'] - box_2['left'] - box_2['right']))

    # 2.) Their left (top) edges align
    row_scores['top_align'] = 1.0 / (1.0 + abs(box_1['top'] - box_2['top']))
    col_scores['left_align'] = 1.0 / (1.0 + abs(box_1['left'] - box_2['left']))

    # 3.) Their right (bottom) edges align
    row_scores['bottom_align'] = 1.0 / (1.0 + abs(box_1['bottom'] - box_2['bottom']))
    col_scores['right_align'] = 1.0 / (1.0 + abs(box_1['right'] - box_2['right']))

    # 4.) If there is a line close to their left (above them)
    # 5.) If there is a line close to their right (below them)
    # 6.) They overlap significantly in their horizontal (vertical) range

    overall_row_scores[str(comb)] = calculate_row_score(row_scores)
    overall_col_scores[str(comb)] = calculate_col_score(col_scores)

  for comb in overall_row_scores:
    print('comb: ' + str(comb))
    print('row score: ' + str(overall_row_scores[comb]))
    print('col score: ' + str(overall_col_scores[comb]))

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
