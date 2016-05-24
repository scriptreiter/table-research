import json
import os
import cv2
import base64
import httplib2

from apiclient.discovery import build
from oauth2client.client import GoogleCredentials

def query_google_ocr(image_content):
  '''Run a label request on a single image'''

  API_DISCOVERY_FILE = 'https://vision.googleapis.com/$discovery/rest?version=v1'
  http = httplib2.Http()

  credentials = GoogleCredentials.get_application_default().create_scoped(
      ['https://www.googleapis.com/auth/cloud-platform'])
  credentials.authorize(http)

  service = build('vision', 'v1', http=http, discoveryServiceUrl=API_DISCOVERY_FILE)

  service_request = service.images().annotate(
    body={
      'requests': [{
        'image': {
          'content': image_content
         },
        'features': [{
          'type': 'TEXT_DETECTION',
          'maxResults': 1
         }]
       }]
    })

  return service_request.execute()

def get_labels(response, combine=False, zoom=1):
  if 'textAnnotations' not in response['responses'][0]:
    return '' if combine else []

  detections = response['responses'][0]['textAnnotations']

  if combine:
    return detections[0]['description'].replace('\n', ' ').strip()
  else:
    return label_boxes(detections[1:], zoom, response['meta'])

def fill_blanks(det, metadata):
  if 'y' not in det['boundingPoly']['vertices'][0]:
    det['boundingPoly']['vertices'][0]['y'] = 0

  if 'y' not in det['boundingPoly']['vertices'][1]:
    det['boundingPoly']['vertices'][1]['y'] = 0

  if 'y' not in det['boundingPoly']['vertices'][2]:
    det['boundingPoly']['vertices'][0]['y'] = metadata['height']

  if 'y' not in det['boundingPoly']['vertices'][3]:
    det['boundingPoly']['vertices'][1]['y'] = metadata['height']

  if 'x' not in det['boundingPoly']['vertices'][0]:
    det['boundingPoly']['vertices'][0]['x'] = 0

  if 'x' not in det['boundingPoly']['vertices'][3]:
    det['boundingPoly']['vertices'][1]['x'] = 0

  if 'x' not in det['boundingPoly']['vertices'][1]:
    det['boundingPoly']['vertices'][0]['x'] = metadata['width']

  if 'x' not in det['boundingPoly']['vertices'][2]:
    det['boundingPoly']['vertices'][1]['x'] = metadata['width']

def label_boxes(detections, zoom, metadata):
  boxes = []
  for det in detections:
    fill_blanks(det, metadata)
    xs = [x['x'] for x in det['boundingPoly']['vertices'] if 'x' in x]
    ys = [x['y'] for x in det['boundingPoly']['vertices'] if 'y' in x]

    min_x = min(xs)
    min_y = min(ys)

    boxes.append(unzoom((min_x, min_y, max(max(xs) - min_x, 1), max(max(ys) - min_y, 1), det['description']), zoom))

  return boxes

def unzoom(box, zoom):
  return (int(box[0] / zoom),
          int(box[1] / zoom),
          int(box[2] / zoom),
          int(box[3] / zoom),
          box[4])

def get_cell_label(cache_base, img_base, photo_file, box, zoom):
  cache_path = cache_base + photo_file + '_' + '_'.join([str(x) for x in box[:4]]) + '.json'

  if os.path.isfile(cache_path):
    with open(cache_path, 'r') as cache_file:
      response = json.loads(cache_file.read())
  else:
    img = cv2.imread(img_base + photo_file)
    x1 = zoom * box[0]
    x2 = x1 + (zoom * box[2])
    y1 = zoom * box[1]
    y2 = y1 + (zoom * box[3])

    cell = img[y1:y2, x1:x2]

    try:
      retval, cell_buffer = cv2.imencode('.jpg', cell)
    except:
      return ''

    image_content = base64.b64encode(cell_buffer).decode()

    response = query_google_ocr(image_content)

    if 'responses' in response:
      with open(cache_path, 'w') as cache_file:
        json.dump(response, cache_file)
    else:
      return ''

  return get_labels(response, combine=True)

def add_labels(boxes, image_base, image_path, cache_path, zoom):
  labeled = []
  for box in boxes:
    label = get_cell_label(cache_path, image_base, image_path, box, zoom)
    labeled.append((box[0], box[1], box[2], box[3], [label]))

  return labeled

def get_image_boxes(cache_base, img_base, photo_file, zoom):
  cache_path = cache_base + photo_file + '.json'

  if os.path.isfile(cache_path):
    with open(cache_path, 'r') as cache_file:
      response = json.loads(cache_file.read())
  else:
    img = cv2.imread(img_base + photo_file)
    height = len(img)
    width = len(img[0])

    try:
      retval, cell_buffer = cv2.imencode('.jpg', img)
    except:
      return []

    image_content = base64.b64encode(cell_buffer).decode()

    response = query_google_ocr(image_content)

    if 'responses' in response:
      response['meta'] = {'width': width, 'height': height}

      with open(cache_path, 'w') as cache_file:
        json.dump(response, cache_file)
    else:
      return []

  # Need to take zoom into account?!
  labels = get_labels(response, combine=False, zoom=zoom)
  return labels
