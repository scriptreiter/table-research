import json
import os
import sub_key

json_cache_path = 'ai2_ocr'

def get_json_data(image, base_path, zoom_level, pref):
  zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''
  json_cache_file = pref + json_cache_path + '/' + zoom_prefix + image + '.json'

  with open(json_cache_file, 'r') as f:
    return json.loads(f.read())

def convert_to_boxes(data, zoom_level):
  boxes = []
  for box in data:
    left = int(box['box'][0] / zoom_level)
    top = int(box['box'][1] / zoom_level)
    width = int(box['box'][2] / zoom_level) - left
    height = int(box['box'][3] / zoom_level) - top
    label = box['text']

    if width == 0 or height == 0:
      print('0x0 box')
      width = 7
      height = 15
      continue

    boxes.append((left, top, width, height, label))

  return boxes
