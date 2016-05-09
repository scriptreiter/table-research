import numpy as np
import cv2
import os

def get_whitespace(img_name, base_path):
  image = get_binary(img_name, base_path)
  markers = [[{} for col in row] for row in image]

  mark_left(markers, image)
  mark_right(markers, image)
  mark_top(markers, image)
  mark_bottom(markers, image)

  threshold = 10

  filter_horiz(markers, threshold)
  filter_vert(markers, threshold)

  col_margins = get_col_margins(markers, threshold)
  row_margins = get_row_margins(markers, threshold)

  return (col_margins, row_margins)

def get_col_margins(markers, threshold):
  margins = []
  for r in range(len(markers) - threshold + 1):
    for c in range(len(markers[r])):
      if 'c_checked' not in markers[r][c]:
        # We want to check this vertical margin
        col = check_col(markers, r, c, threshold)

        if col is not None:
          margins.append(col)

  return margins

def get_row_margins(markers, threshold):
  margins = []
  for r in range(len(markers)):
    for c in range(len(markers[r]) - threshold + 1):
      if 'r_checked' not in markers[r][c]:
        # We want to check this horizontal margin
        row = check_row(markers, r, c, threshold)

        if row is not None:
          margins.append(row)

  return margins

def check_col(markers, r, c, threshold):
  i = 0

  # Increment until we reach a row with an invalid marker
  while r + i < len(markers) and markers[r + i][c]['horiz'] == 255:
    # Mark as checked
    markers[r + i][c]['c_checked'] = True
    i += 1

  # We reached an invalid marker or the end of the image
  if i >= threshold:
    return (r, c, i)
  else:
    return None

def check_row(markers, r, c, threshold):
  i = 0

  # Increment until we reach a row with an invalid marker
  while c + i < len(markers[r]) and markers[r][c + i]['vert'] == 255:
    # Mark as checked
    markers[r][c + i]['r_checked'] = True
    i += 1

  # We reached an invalid marker or the end of the image
  if i >= threshold:
    return (r, c, i)
  else:
    return None

def filter_horiz(markers, threshold):
  for r in range(len(markers)):
    for c in range(len(markers[r])):
      curr = markers[r][c]
      label = 255

      if curr['left'] + curr['right'] < threshold:
        label = 0

      markers[r][c]['horiz'] = label
        

def filter_vert(markers, threshold):
  for r in range(len(markers)):
    for c in range(len(markers[r])):
      curr = markers[r][c]
      label = 255

      if curr['top'] + curr['bottom'] < threshold:
        label = 0

      markers[r][c]['vert'] = label

def get_binary(img_name, base_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

  return bin_img

# Could combine these all, but doing this for the moment to avoid
# errors with making it more generic
def mark_left(markers, image):
  for r in range(len(image)):
    for c in range(len(image[r])):
      if image[r][c] == 0:
        markers[r][c]['left'] = 0
      else:
        if c != 0:
          markers[r][c]['left'] = markers[r][c-1]['left'] + 1
        else:
          markers[r][c]['left'] = float('inf')

def mark_right(markers, image):
  for r in range(len(image) - 1, -1, -1):
    for c in range(len(image[r]) - 1, -1, -1):
      if image[r][c] == 0:
        markers[r][c]['right'] = 0
      else:
        if c != len(image[r]) - 1:
          markers[r][c]['right'] = markers[r][c+1]['right'] + 1
        else:
          markers[r][c]['right'] = float('inf')

def mark_top(markers, image):
  for r in range(len(image)):
    for c in range(len(image[r])):
      if image[r][c] == 0:
        markers[r][c]['top'] = 0
      else:
        if r != 0:
          markers[r][c]['top'] = markers[r-1][c]['top'] + 1
        else:
          markers[r][c]['top'] = float('inf')

def mark_bottom(markers, image):
  for r in range(len(image) - 1, -1, -1):
    for c in range(len(image[r]) - 1, -1, -1):
      if image[r][c] == 0:
        markers[r][c]['bottom'] = 0
      else:
        if r != len(image) - 1:
          markers[r][c]['bottom'] = markers[r+1][c]['bottom'] + 1
        else:
          markers[r][c]['bottom'] = float('inf')

def test():
  base_dir = 'alternate_images'
  images = [img for img in os.listdir(base_dir) if img.endswith('.jpg')]
  for image in images:
    print('working on: ' + image)
    temp = get_whitespace(image, base_dir)

    show_cols(image, base_dir, temp[0])
    show_rows(image, base_dir, temp[1])

def output(markers, mark, name='test_output.png'):
  cv2.imwrite(name, np.array([[y[mark] for y in x] for x in markers]))

def show_cols(image, base_dir, cols):
  img = cv2.imread(base_dir + '/' + image)

  colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (0, 128, 128), (128, 0, 128), (128, 255, 128)]
  
  for i, col in enumerate(cols):
    cv2.line(img, (col[1], col[0]), (col[1], col[0] + col[2]), colors[i % len(colors)], 1)

  cv2.imwrite('temp_out/' + image + '_cols.jpg', img)

def show_rows(image, base_dir, rows):
  img = cv2.imread(base_dir + '/' + image)

  colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (0, 128, 128), (128, 0, 128), (128, 255, 128)]
  
  for i, row in enumerate(rows):
    cv2.line(img, (row[1], row[0]), (row[1] + row[2], row[0]), colors[i % len(colors)], 1)

  cv2.imwrite('temp_out/' + image + '_rows.jpg', img)
