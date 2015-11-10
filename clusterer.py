def cluster_scores(score_matrix, threshold):
  clusters = []
  for i in range(len(score_matrix)):
    curr_cluster = set([i])
    for j in range(len(score_matrix)):
      curr_score = score_matrix[i][j]
      if curr_score > threshold:
        curr_cluster.add(j)

      clusters.append(curr_cluster)

  # Now we need to merge any clusters with shared elements
  # Based roughly on the algorithm from Niklas at:
  # http://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
  have_merged = True
  while have_merged:
    have_merged = False
    new_clusters = []
    while len(clusters) > 0:
      first = clusters[0]
      remaining = clusters[1:]
      clusters = []

      for cluster in remaining:
        if cluster.isdisjoint(first):
          clusters.append(cluster)
        else:
          have_merged = True
          first |= cluster

      new_clusters.append(first)

    clusters = new_clusters

  return clusters

