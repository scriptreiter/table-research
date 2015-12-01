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
import spreadsheeter
import scorer

# Will need to do some interesting work to see if we need an edge line

json_file = 'full_output.json'
json_cache_path = 'json_cache'
full_img = 'api_test_image_full.jpg'
output_file = 'detected_boxes.jpg'
verbose = False
# full_base_dir = 'images/table_training'
full_base_dir = 'regents_table'
img_pref = 'regents/'
xlsx_path = 'xlsx_adjusted'
json_out_path = 'json_out'
zoom_level = 3
sleep_delay = 0

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
    # Set the current image for the evaluation scorer
    scorer.set_current_image(image)

    # if not image.startswith('005'):
      # continue

    print('Processing: ' + image)

    data = oxford_api.get_json_data(image, base_dir, zoom_level, img_pref);

    lines = liner.get_lines(image, base_dir)

    boxes, raw_boxes = boxer.get_boxes(data, zoom_level, lines)

    scores = liner.rate_lines(lines, boxes)

    filtered_lines = liner.filter_lines(lines, boxes, scores);

    new_lines = liner.remove_lines(lines, filtered_lines, scores)

    rows, cols = score_rows.get_structure(boxes, new_lines)

    if verbose:
      print_structure(rows, 'Rows')
      print_structure(cols, 'Cols')

    draw_lines(base_dir + '/' + image, lines, img_pref + 'table_labeling/' + image + '_orig.jpg')
    draw_lines(base_dir + '/' + image, new_lines, img_pref + 'table_labeling/' + image)

    draw_structure(translate_box_paradigm(raw_boxes), base_dir + '/' + image, img_pref + 'table_structure/' + image + '_raw_boxes.jpg')
    draw_structure(translate_box_paradigm(boxes), base_dir + '/' + image, img_pref + 'table_structure/' + image + '_merged_boxes.jpg')
    draw_structure(rows, base_dir + '/' + image, img_pref + 'table_structure/' + image + '_rows.jpg')
    draw_structure(cols, base_dir + '/' + image, img_pref + 'table_structure/' + image + '_cols.jpg')

    zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''
    spreadsheeter.output(rows, cols, boxes, img_pref + xlsx_path + '/' + zoom_prefix + image + '.xlsx', img_pref + json_out_path + '/' + zoom_prefix + image + '.json')

    if verbose:
      print('Estimating (' + str(len(new_lines[0]) - 1) + ' x ' + str(len(new_lines[1]) - 1) + ')')

      print()

    if sleep_delay > 0:
      time.sleep(sleep_delay)

  scorer.evaluate()

def print_structure(clusters, label):
  print('Printing structure (' + label + ')')

  for box in clusters:
    print('box: (' + str(box[0]) + ', ' + str(box[1]) + ', ' + str(box[2]) + ', ' + str(box[3]) + ')')
    print('  text: ' + ', '.join(box[4]))

def translate_box_paradigm(boxes):
  new_boxes = []

  for box in boxes:
    new_boxes.append((box[0], box[1], box[0] + box[2], box[1] + box[3], box[4]))

  return new_boxes

def draw_structure(boxes, img_path, output_img):
  img = cv2.imread(img_path)

  for box in boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

  cv2.imwrite(output_img, img)

def draw_lines(img_path, lines, output_file):
  img = cv2.imread(img_path)
  horiz_lines = lines[0]
  vert_lines = lines[1]

  for line in horiz_lines:
    cv2.line(img, (line['start'], line['border']), (line['end'], line['border']), (0, 0, 255), 1);

  for line in vert_lines:
    cv2.line(img, (line['border'], line['start']), (line['border'], line['end']), (0, 0, 255), 1);

  cv2.imwrite(output_file, img)

if __name__ == '__main__':
  main()
