def get_boxes(data):
  return get_boxes_from_json(data)

def get_boxes_from_json(data):
  boxes = []
  if 'regions' in data:
    for region in data['regions']:
      if 'lines' in region:
        for line in region['lines']:
          if 'words' in line:
            for word in line['words']:
              if 'boundingBox' in word:
                bbox = [int(int(x) / 2) for x in word['boundingBox'].split(',')]
                if 'text' in word:
                  bbox.append(word['text'])
                boxes.append(bbox)

  return boxes
