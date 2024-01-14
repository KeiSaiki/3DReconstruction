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

camera_matrix = np.array([
    [507.67984091855385, 0, 330.27513286880281],
    [0, 509.84029418345722, 242.96535901758909],
    [0, 0, 1]
], dtype="double")

def get_camera_direction(tvec, rvec, u=camera_matrix[0][2], v=camera_matrix[1][2]):
  rmat, _ = cv2.Rodrigues(rvec)
  rR = rmat
  tR = -rmat.T @ tvec
  print("\nt Raw")
  print(tR)
  print("\nr Raw")
  print(rR)
  P = camera_matrix @ np.hstack((rR, tR))
  print("\nP")
  print(P)
  p, d = find_intersection_line( 
    P[0][0] - u*P[2][0],
    P[0][1] - u*P[2][1],
    P[0][2] - u*P[2][2],
    P[0][3] - u,
    P[1][0] - v*P[2][0],
    P[1][1] - v*P[2][1],
    P[1][2] - v*P[2][2],
    P[1][3] - v,
    specified_x=tR[0][0],
  )
  return (np.array(p, dtype="double"), np.array(d, dtype="double"))

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

#

print(find_intersection_line(1, -2, -1, -8, 4, 1, -2, -5, specified_z=0))

#

tvec = np.array([[ -75.62213202],[ 118.76879132],[4329.49507308]], dtype="double")
rvec = np.array([[ 1.77554769],[-0.27088635],[ 0.13907835]], dtype="double")
u, v = (126, 346)
p, d = get_camera_direction(tvec, rvec, u, v)
print("\nposition")
print(p)
print("\ndirection")
print(d)
def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)

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
plot_vector(plt, p, d/10, color="purple")
plt.show()