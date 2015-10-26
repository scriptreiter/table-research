import cv2
import numpy as np
import os

def bin_value(val, bins, tolerance):
  for bin in bins:
    for bin_val in bin:
      if abs(val - bin_val) < tolerance:
        bin.append(val)
        return

  # Not within tolerance of any bin
  bins.append([val])

def display_bins(bins):
  for bin in bins:
    print(bin)

def average_bins(bins):
  averaged_bins = []
  for bin in bins:
    averaged_bins.append(sum(bin) / len(bin))

  return averaged_bins

def process_image(img_name, base_path):
  print('Processing: ' + img_name)

  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # cv2.imwrite('gray.jpg', gray)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
  # cv2.imwrite('edges.jpg', edges)
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
  
  horiz_count = 0
  vert_count = 0
  
  horiz_lines = []
  vert_lines = []
  
  for info in lines:
    rho = info[0][0]
    theta = info[0][1]
  
    if abs(theta - (np.pi / 2)) < 0.1 or abs(theta - (np.pi * 3 / 2)) < 0.1:
      print('This is a horizontal line')
      y = int(np.sin(theta) * rho)
      horiz_lines.append(y)
      horiz_count += 1
    elif abs(theta - 0) < 0.1 or abs(theta - np.pi) < 0.1:
      print('This is a vertical line')
      x = int(np.cos(theta) * rho)
      vert_lines.append(x)
      vert_count += 1
    else:
      print(theta)
  
  # cv2.imwrite('houghlines.jpg', img)
  
  # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
  
  print('horiz:')
  for y in horiz_lines:
    print(y)
  
  print('')
  print(horiz_count)
  print('vert:')
  for x in vert_lines:
    print(x)
  
  print('')
  print(vert_count)
  
  # Now bin the lines
  tolerance = 6
  horiz_bins = []
  vert_bins = []
  
  for y in horiz_lines:
    bin_value(y, horiz_bins, tolerance)
  
  for x in vert_lines:
    bin_value(x, vert_bins, tolerance)
  
  print('horiz bins')
  display_bins(horiz_bins)
  
  print('vert bins')
  display_bins(vert_bins)
  
  # Now average out the bins
  horiz_markers = average_bins(horiz_bins)
  horiz_markers.sort()
  
  vert_markers = average_bins(vert_bins)
  vert_markers.sort()
  
  print('avg horiz')
  display_bins(horiz_markers)
  
  print('avg vert')
  display_bins(vert_markers)
  
  # For additional filtering, we could rely on calculating the grid
  # size, and checking against that, and multiples/fractions
  
  # # Now we need to bin the differences
  # # We can use this to filter out ones that are not along grid lines
  # # But will want to use the actual markers to step, probably
  # grid_tolerance = 3
  # horiz_grid_bins = []
  # vert_grid_bins = []
  # 
  # for idx, val in enumerate(horiz_markers):
  #   if idx != 0:
  #     bin_value(val, horiz_grid_bins, grid_tolerance)
  
  # Now we can iterate through our grid cells, and get bounding boxes
  for i, x in enumerate(vert_markers):
    # Fencepost
    if i == 0:
      continue
  
    for j, y in enumerate(horiz_markers):
      # Fencepost
      if j == 0:
        continue
  
      # We have a valid bounding box, save it
      x1 = vert_markers[i - 1]
      x2 = vert_markers[i]
      y1 = horiz_markers[j - 1]
      y2 = horiz_markers[j]
      grid_cell = img[y1:y2, x1:x2]
  
      cv2.imwrite('cells/' + img_name + '_' + str(i) + '_' + str(j) + '.jpg', grid_cell)

images = os.listdir('basic_test_images')

for image in images:
  process_image(image, 'basic_test_images')
