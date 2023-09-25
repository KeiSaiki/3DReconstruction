import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vector(axes, location, vector, color = "red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color = color, length = 1,
              arrow_length_ratio = 0.1)

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
      (self._x, self._y, self._z),
      (0, self._y, self._z),
    ], dtype = "double")
    return result

image_points = np.array([
  (302, 365),
  (367, 319),
  (212, 342),
  (303, 287),
  (370, 259),
  (292, 247),
  (211, 271),
], dtype = "double")

# Camera Intrinsic Paramter
dist_coeffs = np.array([
  [3.63211720e-01,
  -1.93851026e+00,
  -1.47109243e-04,
  5.95798535e-04,
  3.14452169e+00]
])
camera_matrix = np.array([
  [508.98721069, 0, 331.12232423],
  [0, 511.00672888, 243.52320381],
  [0, 0, 1]
], dtype = "double")

# ---
def makeline(img_x, img_y, cam_tvec, cam_rvec):
  arr = np.array([img_x, img_y, 1])
  R, _ = cv2.Rodrigues(cam_rvec)
  arr = arr * camera_matrix.I * np.hstack([R, cam_tvec]).I
  return np.delete(arr, 3)

def deteremime_point(p1, d1, p2, d2):
  result = np.array([d1*d1, -d1*d2], [d1*d2, -d2*d2]).I * np.array([(p2-p1)*d1, (p2-p1)*d2])
  return result

# --------------------------------------

origin = ObjectBox(348, 469, 238)

print("Solve Perspective N Point Problem")

print("\nCamera Matrix : ")
print(camera_matrix)

print("\nDistortion Coefficient")
print(dist_coeffs)

(success, rvec, tvec) = cv2.solvePnP(origin.object_points(), image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
print("\nTranslation Vector")
print(tvec)

print("\nRotation Vector")
print(rvec)

print("\nRotation Matrix")
R, jacob = cv2.Rodrigues(rvec)
print(R)

# 3Dグラフ描画部------------------
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.grid()
ax.set_xlabel("x [m]", fontsize = 12)
ax.set_ylabel("y [m]", fontsize = 12)
ax.set_zlabel("z [m]", fontsize = 12)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 3)
plot_vector(plt, [0, 0, 0], [origin._x/1000, 0, 0], color = "red")
plot_vector(plt, [0, 0, 0], [0, origin._y/1000, 0], color = "green")
plot_vector(plt, [0, 0, 0], [0, 0, origin._z/1000], color = "blue")
plot_vector(plt, [0, 0, 0], R*tvec/1000, color = "black")
plt.show()

# ref: https://daily-tech.hatenablog.com/entry/2018/02/02/071655
#      https://python.atelierkobato.com/quiver
