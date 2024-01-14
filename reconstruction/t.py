import numpy as np

def midpoint_of_common_perpendicular(line1, line2):
    p1, v1 = line1
    p2, v2 = line2

    p1, v1, p2, v2 = map(np.array, [p1, v1, p2, v2])

    v1xv2 = np.cross(v1, v2)

    t = np.dot(np.cross(p2 - p1, v2), v1xv2) / np.linalg.norm(v1xv2)**2
    s = np.dot(np.cross(p2 - p1, v1), v1xv2) / np.linalg.norm(v1xv2)**2

    point_on_line1 = p1 + t * v1
    point_on_line2 = p2 + s * v2
    midpoint = (point_on_line1 + point_on_line2) / 2

    return midpoint

#
pa1 = [2, 0, 0]
pa2 = [-1, 1, 0]
pa3 = [-1, -1, 0]
pb1 = [2, 0, 10]
pb2 = [-1, 1, 10]
pb3 = [-1, -1, 10]
pa1, pa2, pa3, pb1, pb2, pb3 = map(np.array, [pa1, pa2, pa3, pb1, pb2, pb3])
lines = [(pa1, pb2 - pa1), (pa2, pb3 - pa2), (pa3, pb1 - pa3)]
  
prob_points = []
for i in range(len(lines)):
  for j in range(i + 1, len(lines)):
    line1, line2 = lines[i], lines[j]
    prob_points.append(midpoint_of_common_perpendicular(line1, line2))

prob_point = [0, 0, 0]
for p in prob_points:
  prob_point = prob_point + p
prob_point = prob_point / len(prob_points)

print(prob_point)