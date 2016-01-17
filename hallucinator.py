import numpy as np
import cv2
from itertools import combinations

def get_contours(img_name, base_path, output_path):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # cv2.imwrite('regents/canny/' + img_name + '_gray.jpg', gray)
  # edges = cv2.Canny(gray, 30, 80, apertureSize = 5)
  bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
  # cv2.imwrite('regents/canny/' + img_name + '_adj.jpg', edges)

  (edges, contours, hierarchy) = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  rects = []
  approxes = []

  for i,contour in enumerate(contours):
    perim = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, perim * 0.02, True)
    approxes.append(approx)

    # If approximated with a quadrilateral, we want to save
    # and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt)
    # 

    if cv2.contourArea(approx) > 200:
      pass
      # import pdb;pdb.set_trace()
      # print('x')
      # Going to need to check if it detects rectangles with multiple cells, and cut those elsewhere
      # and use them for table label info like title, etc.

    if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 200:
      temp_contour = approx.reshape(-1, 2)
      max_cos = np.max([angle_cos(temp_contour[i], temp_contour[(i+1) % 4], temp_contour[(i+2) % 4]) for i in range(4)])

      if max_cos < 0.1:
        rects.append((i,approx)) # Keep the index and the contour

  mark_contours(img, [c for (i, c) in rects], output_path, 'rects', True)
  mark_contours(img, contours, output_path, 'all_contours', True)
  mark_contours(img, approxes, output_path, 'all_approx', True)

  return (rects, hierarchy)

def get_child_contours(rects, hierarchy):
  idxes = [i for (i, c) in rects]

  # Select the contours in this group that are child contours
  # These should be cell-level contours, hopefully
  return [contour for (idx, contour) in rects if no_children(idxes, hierarchy, idx)]

def no_children(idxes, hierarchy, idx):
  children = get_children(hierarchy, idx)

  while len(children) > 0:
    curr_idx = children.pop()

    # If the current child is in the list
    # Then we can shortcut and exit
    if curr_idx in idxes:
      return False

    # Get any children of the current child
    children += get_children(hierarchy, curr_idx)

  return True

def get_children(hierarchy, idx):
  return [i for i,x in enumerate(hierarchy[0]) if x[3] == idx]

def get_root_contours(rects, hierarchy):

  # Select the contour in this group with no parents
  # Hopefully there is only one, and this is the whole image
  parent_idx = [idx for (idx, contour) in rects if hierarchy[0][idx][3] == -1][0]

  # Now want to select the ones that are a direct child of this one
  # Biggest is likely the table
  root_rects = [(idx, contour) for (idx, contour) in rects if hierarchy[0][idx][3] == parent_idx]

  return root_rects

def get_largest_contour(contours):
  max_area = 0.0
  max_contour = contours[0]

  for contour in contours:
    area = cv2.contourArea(contour)

    if area > max_area:
      max_area = area
      max_contour = contour

  return max_contour

def angle_cos(p0, p1, p2):
  d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
  return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def mark_contours(img, contours, output_path, ext, diff_colors=False):
  colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (0, 128, 128), (128, 0, 128), (128, 255, 128)]
  color_idx = 0

  img_copy = img.copy()

  if not diff_colors:
    cv2.drawContours(img_copy, contours, -1, colors[color_idx], 1)
  else:
    for contour in contours:
      cv2.drawContours(img_copy, [contour], 0, colors[color_idx % len(colors)], 3)
      color_idx += 1

  cv2.imwrite(output_path + '_' + ext + '.jpg', img_copy)

def contours_to_boxes(contours):
  boxes = []
  for contour in contours:
    max_x = max_y = float('-inf')
    min_x = min_y = float('inf')

    for point in contour:
      max_x = max(max_x, point[0][0])
      max_y = max(max_y, point[0][1])
      min_x = min(min_x, point[0][0])
      min_y = min(min_y, point[0][1])

    # Storing as x, y, width, height)
    boxes.append((min_x, min_y, max_x - min_x, max_y - min_y, ''))

  return boxes
