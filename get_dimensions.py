import json
import numpy as np
import os
import cv2
from itertools import combinations
import http.client, urllib.request, urllib.parse, urllib.error, base64

import score_rows
import sub_key

# Will need to do some interesting work to see if we need an edge line

# API vars
headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': sub_key.get_key(),
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'unk',
    'detectOrientation ': 'true',
})

json_file = 'full_output.json'
json_cache_path = 'json_cache'
full_img = 'api_test_image_full.jpg'
output_file = 'detected_boxes.jpg'
verbose = True
full_base_dir = 'images/table_training'
should_run_full_test = False

def main():
  if should_run_full_test:
    print('Running full test')
    run_full_test()
  else:
    print('Running single test')
    run_single_test()

def run_full_test():
  images = [img for img in os.listdir(full_base_dir) if img.endswith('.jpg')]
  run_test(images, full_base_dir)

def run_single_test():
  run_test([full_img], '.')
  
def run_test(images, base_dir):
  for image in images:
    print('Processing: ' + image)

    data = get_json_data(image, base_dir);

    boxes = get_boxes_from_file(data)

    lines = get_lines(image, base_dir)

    scores = rate_lines(lines, boxes)

    filtered_lines = filter_lines(lines, boxes, scores);

    new_lines = remove_lines(lines, filtered_lines, scores)

    rows, columns = score_rows.get_structure(boxes, new_lines)

    draw_lines(base_dir + '/' + image, new_lines, 'images/table_labeling/' + image)

    print('Estimating (' + str(len(new_lines[0]) - 1) + ' x ' + str(len(new_lines[1]) - 1) + ')')

def get_json_data(image, base_path):
  json_cache_file = json_cache_path + '/' + image + '.json'

  if os.path.isfile(json_cache_file):
    return json.loads(open(json_cache_file, 'r').read())

  img_data = open(base_path + '/' + image, 'rb').read()

  try:
    conn = http.client.HTTPSConnection('api.projectoxford.ai')
    conn.request("POST", "/vision/v1/ocr?%s" % params, img_data, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
  except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

  json_data = json.loads(data.decode('utf-8')) # Need to double-check if utf-8 is correct

  with open(json_cache_file, 'w') as json_file:
    json.dump(json_data, json_file)

  return json_data

# Could remove the negative scoring lines beforehand...
# Or change the scores of the filtered lines to be negative
def remove_lines(lines, filtered_lines, scores):
  new_horiz_lines = cut_lines(lines[0], filtered_lines[0], scores[0])
  new_vert_lines = cut_lines(lines[1], filtered_lines[1], scores[1])

  return (new_horiz_lines, new_vert_lines)

def cut_lines(lines, filtered, scores):
  new_lines = []
  for i, line in enumerate(lines):
    # Could do this more efficiently, particularly since it's sorted
    if i not in filtered and scores[i] > 0:
      new_lines.append(line)

  return new_lines

def filter_lines(lines, boxes, scores):
  horiz_lines = lines[0]
  vert_lines = lines[1]

  horiz_scores = scores[0]
  vert_scores = scores[1]

  horiz_removed_lines = check_lines(horiz_lines, boxes, horiz_scores, 1)
  vert_removed_lines = check_lines(vert_lines, boxes, vert_scores, 0)

  if verbose:
    for line_i in horiz_removed_lines:
      print('Removed h line #' + str(line_i) + ' at ' + str(horiz_lines[line_i]));

    for line_i in vert_removed_lines:
      print('Removed v line #' + str(line_i) + ' at ' + str(vert_lines[line_i]));

  return (horiz_removed_lines, vert_removed_lines)

def check_lines(lines, boxes, scores, offset):
  removed_lines = set()
  for comb in combinations(enumerate(lines), 2):
    min_val = min(comb[0][1], comb[1][1])
    max_val = max(comb[0][1], comb[1][1])

    box_inbetween = False
    for box in boxes:
      box_edge_1 = box[offset]
      box_edge_2 = box[offset] + box[offset + 2]
      if (box_edge_1 < max_val and box_edge_1 > min_val) or (box_edge_2 < max_val and box_edge_2 > min_val):
        box_inbetween = True

    if not box_inbetween:
      # We need to choose one line over the other
      if scores[comb[0][0]] > scores[comb[1][0]]:
        removed_lines.add(comb[1][0])
      else:
        removed_lines.add(comb[0][0])

  return removed_lines

def rate_lines(lines, boxes):
  horiz_lines = lines[0]
  vert_lines = lines[1]
  horiz_lines.sort()
  vert_lines.sort()

  horiz_scores = calc_line_box_scores(horiz_lines, boxes, 1)
  vert_scores = calc_line_box_scores(vert_lines, boxes, 0)

  if verbose:
    for i, line in enumerate(horiz_lines):
      print('H Line at ' + str(line) + ' with a score of ' + str(horiz_scores[i]))

    for i, line in enumerate(vert_lines):
      print('V Line at ' + str(line) + ' with a score of ' + str(vert_scores[i]))

  return (horiz_scores, vert_scores)

def calc_line_box_scores(lines, boxes, offset):
  scores = []

  for line in lines:
    min_margin = float('inf')

    num_intersections = 0
    intersection_penalty = 0

    for box in boxes:
      first_edge = box[offset]
      second_edge = box[offset] + box[offset + 2]

      # Calculate the minimum distance to either edge of this box
      min_to_edge = min(abs(line - first_edge), abs(line - second_edge))

      # If it intersections the box, track that, and penalize based
      # on how far into the box it is
      if line >= first_edge and line <= second_edge:
        num_intersections += 1
        intersection_penalty += min_to_edge

      # Could track some sense of uniformity in the closest edges
      # Although the case of this provides a problem:
      #
      # Line 1
      # Line 2        Line 1      Line 1
      # Line 3
      #
      # Because it would preference grid lines through cell 1

      # Check if this is the smallest margin, yet
      if min_to_edge < min_margin:
        min_margin = min_to_edge

    score = min_margin - (num_intersections * intersection_penalty)

    scores.append(score)

  return scores

def get_boxes_from_file(data):
  boxes = []
  if 'regions' in data:
    for region in data['regions']:
      if 'lines' in region:
        for line in region['lines']:
          if 'words' in line:
            for word in line['words']:
              if 'boundingBox' in word:
                bbox = [int(x) for x in word['boundingBox'].split(',')]
                if 'text' in word:
                  bbox.append(word['text'])
                boxes.append(bbox)

  return boxes

def bin_value(val, bins, tolerance):
  for bin in bins:
    for bin_val in bin:
      if abs(val - bin_val) < tolerance:
        bin.append(val)
        return

  # Not within tolerance of any bin
  bins.append([val])

def display_bins(bins):
  if verbose:
    for bin in bins:
      print(bin)

def average_bins(bins):
  averaged_bins = []
  for bin in bins:
    averaged_bins.append(sum(bin) / len(bin))

  return averaged_bins

def get_lines(img_name, base_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)

  if lines is None:
    lines = []
  
  horiz_count = 0
  vert_count = 0
  
  horiz_lines = []
  vert_lines = []
  
  for info in lines:
    rho = info[0][0]
    theta = info[0][1]
  
    if abs(theta - (np.pi / 2)) < 0.1 or abs(theta - (np.pi * 3 / 2)) < 0.1:
      # This is a horizontal line
      y = int(np.sin(theta) * rho)
      horiz_lines.append(y)
      horiz_count += 1
    elif abs(theta - 0) < 0.1 or abs(theta - np.pi) < 0.1:
      # This is a vertical line
      x = int(np.cos(theta) * rho)
      vert_lines.append(x)
      vert_count += 1
    elif verbose:
      print('Nonstandard line: ' + str(theta))

  return (horiz_lines, vert_lines)
  
def get_sorted_avg_lines(lines):
  horiz_lines = lines[0]
  vert_lines = lines[1]

  # Now bin the lines
  tolerance = 6
  horiz_bins = []
  vert_bins = []
  
  for y in horiz_lines:
    bin_value(y, horiz_bins, tolerance)
  
  for x in vert_lines:
    bin_value(x, vert_bins, tolerance)
  
  # Now average out the bins
  horiz_markers = average_bins(horiz_bins)
  horiz_markers.sort()
  
  vert_markers = average_bins(vert_bins)
  vert_markers.sort()

  return (horiz_markers, vert_markers)

def draw_lines(img_path, lines, output_file):
  img = cv2.imread(img_path)
  horiz_lines = lines[0]
  vert_lines = lines[1]

  height, width, channels = img.shape

  for line in horiz_lines:
    cv2.line(img, (0, line), (width, line), (0, 255, 0), 1);

  for line in vert_lines:
    cv2.line(img, (line, 0), (line, height), (0, 255, 0), 1);

  cv2.imwrite(output_file, img)

if __name__ == '__main__':
  main()
