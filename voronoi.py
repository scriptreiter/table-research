from scipy.spatial import Voronoi
import spacer

import cv2
import os

import numpy as np

def get_voronoi(img_name, base_path):
  image = spacer.get_binary(img_name, base_path)

  points = get_black_points(image)

  return Voronoi(points)

def get_black_points(image):
  points = []

  for r in range(len(image)):
    for c in range(len(image[r])):
      if image[r][c] == 0:
        points.append([c, r]) # Need to flip?

  return points

def process_image_points(image, base_dir, output_dir, points):
  try:
    vinfo = Voronoi(points)
    output_info(image, base_dir, output_dir, vinfo)
  except:
    print('  Voronoi failed for image: ' + image)

def test():
  base_dir = 'vtest'
  images = [img for img in os.listdir(base_dir)]# if img.endswith('.jpg')]
  for image in images:
    print('working on: ' + image)
    vinfo = get_voronoi(image, base_dir)
    output(image, base_dir, vinfo)

def output(image, base_dir, vinfo):
  output_info(image, base_dir, 'temp_v_out/', vinfo)

def output_info(image, base_dir, output_dir, vinfo):
  img = cv2.imread(base_dir + '/' + image)
  img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)

  for pts, center_pts in zip(vinfo.ridge_vertices, vinfo.ridge_points):
    if pts[0] == -1 or pts[1] == -1:
      p1, p2 = get_unbounded_point(vinfo, center_pts[0], center_pts[1], pts[0], pts[1])
    else:
      p1 = vinfo.vertices[pts[0]]
      p2 = vinfo.vertices[pts[1]]

    p1 = transform_point(p1)
    p2 = transform_point(p2)
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)

  cv2.imwrite(output_dir + image, img)

def transform_point(pt):
  return [int(2 * x) for x in pt]

# Adaprted from: https://gist.github.com/pv/8036995 and
# http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def get_unbounded_point(vinfo, p1, p2, v1, v2):
  if v2 < 0:
    v1, v2 = v2, v1

  t = vinfo.points[p2] - vinfo.points[p1]
  t /= np.linalg.norm(t)
  n = np.array([-t[1], t[0]])

  center = vinfo.points.mean(axis=0)
  radius = vinfo.points.ptp(axis=0).max()*2

  midpoint = vinfo.points[[p1, p2]].mean(axis=0)
  direction = np.sign(np.dot(midpoint - center, n)) * n
  far_point = vinfo.vertices[v2] + direction * radius

  return [far_point, vinfo.vertices[v2]]

def voronoi_finite_vertices(vor, radius=None):
  """
  Reconstruct infinite voronoi regions in a 2D diagram to finite
  regions.
  Parameters
  ----------
  vor : Voronoi
    Input diagram
  radius : float, optional
    Distance to 'points at infinity'.
  Returns
  -------
  regions : list of tuples
    Indices of vertices in each revised Voronoi regions.
  vertices : list of tuples
    Coordinates for revised Voronoi vertices. Same as coordinates
    of input vertices, with 'points at infinity' appended to the
    end.
  """

  if vor.points.shape[1] != 2:
    raise ValueError("Requires 2D input")

  new_regions = []
  new_vertices = vor.vertices.tolist()

  center = vor.points.mean(axis=0)
  if radius is None:
    radius = vor.points.ptp().max()*2

  # Construct a map containing all ridges for a given point
  all_ridges = {}
  for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
    all_ridges.setdefault(p1, []).append((p2, v1, v2))
    all_ridges.setdefault(p2, []).append((p1, v1, v2))

  # Reconstruct infinite regions
  for p1, region in enumerate(vor.point_region):
    vertices = vor.regions[region]

    if all(v >= 0 for v in vertices):
      # finite region
      new_regions.append(vertices)
      continue

    # reconstruct a non-finite region
    ridges = all_ridges[p1]
    new_region = [v for v in vertices if v >= 0]

    for p2, v1, v2 in ridges:
      if v2 < 0:
        v1, v2 = v2, v1
      if v1 >= 0:
        # finite ridge: already in the region
        continue

      # Compute the missing endpoint of an infinite ridge

      t = vor.points[p2] - vor.points[p1] # tangent
      t /= np.linalg.norm(t)
      n = np.array([-t[1], t[0]])  # normal

      midpoint = vor.points[[p1, p2]].mean(axis=0)
      direction = np.sign(np.dot(midpoint - center, n)) * n
      far_point = vor.vertices[v2] + direction * radius

      new_region.append(len(new_vertices))
      new_vertices.append(far_point.tolist())

    # sort region counterclockwise
    vs = np.asarray([new_vertices[v] for v in new_region])
    c = vs.mean(axis=0)
    angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
    new_region = np.array(new_region)[np.argsort(angles)]

    # finish
    new_regions.append(new_region.tolist())

  return new_regions, np.asarray(new_vertices)
