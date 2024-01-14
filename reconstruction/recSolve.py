import numpy as np
import cv2
import os
import math
import sympy
from sympy import symbols, Eq, solve
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import initialSetup as istu
import recSetup as stu

def get_camera_direction(tvec, rvec, u = istu.camera_matrix[0][2], v = istu.camera_matrix[1][2]):
  rmat, _ = cv2.Rodrigues(rvec)
  r_raw = rmat
  t_raw = -rmat.T @ tvec
  P = istu.camera_matrix @ np.hstack((r_raw, t_raw))
  _, d = find_intersection_line(
    P[0][0] - u*P[2][0],
    P[0][1] - u*P[2][1],
    P[0][2] - u*P[2][2],
    P[0][3] - u,
    P[1][0] - v*P[2][0],
    P[1][1] - v*P[2][1],
    P[1][2] - v*P[2][2],
    P[1][3] - v,
  )
  return (np.array([t_raw[0][0], t_raw[1][0], t_raw[2][0]]), np.array(d))

def find_intersection_line(a1, b1, c1, d1, a2, b2, c2, d2, specified_x=None, specified_y=None, specified_z=None):
  x, y, z = symbols('x y z') # 定義された変数

  plane1 = Eq(a1*x + b1*y + c1*z + d1, 0) # 平面の方程式
  plane2 = Eq(a2*x + b2*y + c2*z + d2, 0)

  direction_vector = (b1*c2 - c1*b2, c1*a2 - a1*c2, a1*b2 - b1*a2) # 方向ベクトルを計算（外積）

  # 交点を求める
  # 引数オプションで指定された値に応じて、どの変数を固定するかを決定
  if specified_x is not None:
    intersection_point = solve([plane1.subs(x, specified_x), plane2.subs(x, specified_x)], (y, z))
    if intersection_point:
      point_vector = (specified_x, intersection_point[y], intersection_point[z])
    else:
      point_vector = None
  elif specified_y is not None:
    intersection_point = solve([plane1.subs(y, specified_y), plane2.subs(y, specified_y)], (x, z))
    if intersection_point:
      point_vector = (intersection_point[x], specified_y, intersection_point[z])
    else:
      point_vector = None
  elif specified_z is not None:
    intersection_point = solve([plane1.subs(z, specified_z), plane2.subs(z, specified_z)], (x, y))
    if intersection_point:
      point_vector = (intersection_point[x], intersection_point[y], specified_z)
    else:
      point_vector = None
  else:
    intersection_point = solve([plane1.subs(x, 0), plane2.subs(x, 0)], (y, z)) # どの変数も指定されていない場合は、x = 0 と仮定
    if intersection_point:
      point_vector = (0, intersection_point[y], intersection_point[z])
    else:
      point_vector = None

  return point_vector, direction_vector

def midpoint_of_common_perpendicular(line1, line2):
    # Extracting position and direction vectors
    p1, v1 = line1
    p2, v2 = line2

    # Converting to numpy arrays
    p1, v1, p2, v2 = map(np.array, [p1, v1, p2, v2])

    # Calculating the cross product of direction vectors
    v1xv2 = np.cross(v1, v2)

    # Calculating coefficients for the parametric equations of the lines
    t = np.dot(np.cross(p2 - p1, v2), v1xv2) / np.linalg.norm(v1xv2)**2
    s = np.dot(np.cross(p2 - p1, v1), v1xv2) / np.linalg.norm(v1xv2)**2

    # Calculating the points on each line
    point_on_line1 = p1 + t * v1
    point_on_line2 = p2 + s * v2

    # Calculating the midpoint of the segment connecting these points
    midpoint = (point_on_line1 + point_on_line2) / 2

    return midpoint

def click_event(event, x, y, flags, param): # マウスクリックイベントをハンドルする
  if event == cv2.EVENT_LBUTTONDOWN:
    clicked_points.append((x, y))
    print(f"Clicked at: ({x}, {y})")

def show_images_and_get_click():
    for i, img in enumerate(stu.images):
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', click_event, i)
        print("\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)
#

clicked_points = []
show_images_and_get_click()

lines = []
for i, (tvec, rvec) in enumerate(stu.camera_poses):
  ui, vi = clicked_points[i]
  lines.append(get_camera_direction(tvec, rvec, ui, vi))
  
prob_points = []
for i in range(len(lines)):
  for j in range(i + 1, len(lines)):
    line1, line2 = lines[i], lines[j]
    prob_points.append(midpoint_of_common_perpendicular(line1, line2))

prob_point = np.array([0, 0, 0], dtype="double")
for p in prob_points:
  prob_point = prob_point + p
  
# prob_point = prob_point / len(prob_points)
print("\nclicked points")
print(clicked_points)
print("\nlines")
print(lines)
print("\nprob_points")
print(prob_points)
print("\nprob_point")
print(prob_point)

fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.grid()
ax.set_xlabel("x [mm]", fontsize=12)
ax.set_ylabel("y [mm]", fontsize=12)
ax.set_zlabel("z [mm]", fontsize=12)
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 2000)
innerx, innery, innerz = (469, 348, 238) # 後で直す。
plot_vector(ax, [0, 0, 0], [innerx, 0, 0], color="red")
plot_vector(ax, [0, 0, 0], [0, innery, 0], color="green")
plot_vector(ax, [0, 0, 0], [0, 0, innerz], color="blue")
for i, (p, d) in enumerate(lines):
  plot_vector(plt, p, d/1000, color="purple")
  v = np.array(p)
  ax.text(v[0], v[1], v[2], f"{i}", color="black")
plt.show()