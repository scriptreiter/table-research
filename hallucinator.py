import numpy as np
import cv2
from itertools import combinations

def get_boxes(img_name, base_path, output_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('regents/canny/' + img_name + '_gray.jpg', gray)
  edges = cv2.Canny(gray, 30, 80, apertureSize = 5)
  cv2.imwrite('regents/canny/' + img_name + '_adj.jpg', edges)

  (edges, contours, hierarchy) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  rects = []
  approxes = []

  for contour in contours:
    perim = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, perim * 0.02, True)
    approxes.append(approx)

    # If approximated with a quadrilateral, we want to save
    # and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt)
    # 

    if cv2.contourArea(approx) > 200:
      # import pdb;pdb.set_trace()
      print('x')
      # Going to need to check if it detects rectangles with multiple cells, and cut those elsewhere
      # and use them for table label info like title, etc.

    if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 200:
      temp_contour = approx.reshape(-1, 2)
      max_cos = np.max([angle_cos(temp_contour[i], temp_contour[(i+1) % 4], temp_contour[(i+2) % 4]) for i in range(4)])

      if max_cos < 0.1:
        rects.append(approx)

  mark_contours(img, rects, output_path, 'rects', True)
  mark_contours(img, contours, output_path, 'all_contours', True)
  mark_contours(img, approxes, output_path, 'all_approx', True)

  return rects

def angle_cos(p0, p1, p2):
  d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
  return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def mark_contours(img, contours, output_path, ext, diff_colors=False):
  colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
  color_idx = 0

  img_copy = img.copy()

  if not diff_colors:
    cv2.drawContours(img_copy, contours, -1, colors[color_idx], 1)
  else:
    for contour in contours:
      cv2.drawContours(img_copy, [contour], 0, colors[color_idx % len(colors)], 3)
      color_idx += 1

  cv2.imwrite(output_path + '_' + ext + '.jpg', img_copy)
