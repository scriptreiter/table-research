import json
import os
import cv2

import get_dimensions

json_cache_path = 'json_cache'
full_base_dir = 'images/table_training'
output_path = 'images/boxes'

def process_image(image_path, data):
  img = cv2.imread(full_base_dir + '/' + image_path)

  if 'regions' in data:
    for region in data['regions']:
      box = [int(x) for x in region['boundingBox'].split(',')]
      cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 255))

      if 'lines' in region:
        for line in region['lines']:
          box = [int(x) for x in line['boundingBox'].split(',')]
          cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 255))

          if 'words' in line:
            for word in line['words']:
              box = [int(x) for x in word['boundingBox'].split(',')]
              cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255))

  cv2.imwrite(output_path + '/' + image_path, img)

def run_full_test():
  images = [img for img in os.listdir(full_base_dir) if img.endswith('.jpg')]
  run_test(images, full_base_dir)
  
def run_test(images, base_dir):
  for image in images:
    data = get_dimensions.get_json_data(image, base_dir);
    process_image(image, data)

if __name__ == '__main__':
  run_full_test()
