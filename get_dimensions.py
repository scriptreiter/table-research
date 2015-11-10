import json
import numpy as np
import os
import sys
import cv2
from itertools import combinations
import http.client, urllib.request, urllib.parse, urllib.error, base64
import time

import score_rows
import sub_key
import oxford_api
import boxer
import liner

# Will need to do some interesting work to see if we need an edge line

json_file = 'full_output.json'
json_cache_path = 'json_cache'
full_img = 'api_test_image_full.jpg'
output_file = 'detected_boxes.jpg'
verbose = False
full_base_dir = 'images/table_training'
zoom_level = 3
sleep_delay = 5

def main():
  if len(sys.argv) > 1:
    img_name = sys.argv[1]

  should_run_full_test = len(sys.argv) < 2

  if should_run_full_test:
    print('Running full test')
    run_full_test()
  else:
    print('Running single test')
    run_single_test(img_name)

def run_full_test():
  images = [img for img in os.listdir(full_base_dir) if img.endswith('.jpg')]
  run_test(images, full_base_dir)

def run_single_test(img_name = full_img):
  run_test([img_name], '.')
  
def run_test(images, base_dir):
  for image in images:
    if not image.startswith('005'):
      continue
    # time.sleep(5)
    print('Processing: ' + image)

    data = oxford_api.get_json_data(image, base_dir, zoom_level);

    lines = liner.get_lines(image, base_dir)

    boxes = boxer.get_boxes(data, zoom_level, lines)

    scores = liner.rate_lines(lines, boxes)

    filtered_lines = liner.filter_lines(lines, boxes, scores);

    new_lines = liner.remove_lines(lines, filtered_lines, scores)

    rows, cols = score_rows.get_structure(boxes, new_lines)

    print_structure(rows, 'Rows')
    print_structure(cols, 'Cols')

    draw_lines(base_dir + '/' + image, new_lines, 'images/table_labeling/' + image)

    draw_structure(rows, cols, base_dir + '/' + image, new_lines, 'images/table_structure/' + image)

    print('Estimating (' + str(len(new_lines[0]) - 1) + ' x ' + str(len(new_lines[1]) - 1) + ')')

    print()

    if sleep_delay > 0:
      time.sleep(sleep_delay)

def print_structure(clusters, label):
  print('Printing structure (' + label + ')')

  for box in clusters:
    print('box: (' + str(box[0]) + ', ' + str(box[1]) + ', ' + str(box[2]) + ', ' + str(box[3]) + ')')
    print('  text: ' + ', '.join(box[4]))

def draw_structure(rows, cols, img_path, lines, output_file):
  img = cv2.imread(img_path)

  for box in rows:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

  for box in cols:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

  cv2.imwrite(output_file, img)

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
