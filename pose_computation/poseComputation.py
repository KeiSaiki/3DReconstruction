import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)

class ObjectBox:
  def __init__(self):
    self._x = 0
    self._y = 0
    self._z = 0
  
  def __init__(self, x, y, z):
    self._x = x
    self._y = y
    self._z = z
  
  def object_points(self):
    result = np.array([
      (0, 0, 0),
      (self._x, 0, 0),
      (0, self._y, 0),
      (0, 0, self._z),
      (self._x, 0, self._z),
      (0, self._y, self._z),
      (self._x, self._y, self._z),
    ], dtype="double")
    return result
  
'''
in1
image_points = np.array([
  (356, 251),
  (422, 198),
  (295, 220),
  (354, 203),
  (423, 155),
  (289, 176),
  (361, 132),
], dtype="double")
'''
'''
in2
image_points = np.array([
  (349, 322),
  (474, 226),
  (265, 238),
  (352, 262),
  (493, 164),
  (255, 178),
  (390, 104),
], dtype="double")
'''
'''
in3
'''
image_points = np.array([
  (358, 275),
  (407, 259),
  (322, 264),
  (357, 240),
  (407, 225),
  (321, 229),
  (371, 216),
], dtype="double")


# Camera Intrinsic Paramter
dist_coeffs = np.array([
  [3.5503921110220771e-01,
  -1.8089641384590791e+00,
  -1.3224112624431249e-03,
  -1.1764534809794213e-03,
  2.7810334600309656e+00]
])
camera_matrix = np.array([
  [507.67984091855385, 0, 330.27513286880281],
  [0, 509.84029418345722, 242.96535901758909],
  [0, 0, 1]
], dtype="double")
'''
fov = 80
pw = 672
ph = 504
fx = 1.0 / (2.0*math.tan(np.radians(fov)/2.0))*pw
fy = fx
cx = pw/2.0
cy = ph/2.0
camera_matrix = np.array([
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1],
])
dist_coeffs = np.zeros((5, 1))
'''

origin = ObjectBox(469, 348, 238)

print("Solve Perspective N Point Problem")

print("\nCamera Matrix")
print(camera_matrix)

print("\nDistortion Coefficient")
print(dist_coeffs)

(success, rvec, tvec) = cv2.solvePnP(origin.object_points(), image_points, camera_matrix, dist_coeffs)
print("\nTranslation Vector")
print(tvec)

print("\nRotation Vector")
print(rvec)

print("\nRotation Matrix")
R_mat, _ = cv2.Rodrigues(rvec)
print(R_mat)

R_raw = R_mat.T
t_raw = -R_mat.T @ tvec
print("\nt_raw")
print(t_raw)

# 3Dグラフ描画部------------------
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.grid()
ax.set_xlabel("x [mm]", fontsize=12)
ax.set_ylabel("y [mm]", fontsize=12)
ax.set_zlabel("z [mm]", fontsize=12)
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 2000)
plot_vector(plt, [0, 0, 0], [origin._x, 0, 0], color="red")
plot_vector(plt, [0, 0, 0], [0, origin._y, 0], color="green")
plot_vector(plt, [0, 0, 0], [0, 0, origin._z], color="blue")
plot_vector(plt, [0, 0, 0], t_raw, color="black")
plt.show()

# ref: https://daily-tech.hatenablog.com/entry/2018/02/02/071655
#      https://python.atelierkobato.com/quiver
#      https://dev.classmethod.jp/articles/estimate-camera-external-parameter-matrix-2d-camera/