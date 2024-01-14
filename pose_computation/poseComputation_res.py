import numpy as np
import cv2
import os
import math
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
from sympy import symbols, Eq, solve

def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)

class ObjectBox:
  def __init__(self, x=0, y=0, z=0):
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

# 画像上のある一点の存在しうる範囲についての関数群 

def get_camera_direction(camera_matrix, tvec, rvec):
  rmat, _ = cv2.Rodrigues(rvec)
  r_raw = rmat
  t_raw = -rmat.T @ tvec
  P = camera_matrix @ np.hstack((r_raw, t_raw))
  u = camera_matrix[0][2]
  v = camera_matrix[1][2]
  d, _ = find_intersection_line(
    P[0][0] - u*P[2][0],
    P[0][1] - u*P[2][1],
    P[0][2] - u*P[2][2],
    P[0][3] - u,
    P[1][0] - v*P[2][0],
    P[1][1] - v*P[2][1],
    P[1][2] - v*P[2][2],
    P[1][3] - v,
  )
  return np.array(d)

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

  return direction_vector, point_vector

# 

class Perspactive:
  def __init__(self):
    self.folder = "/Users/keisaiki/Documents/Lab/3DReconstruction/pose_computation/images"
    self.images = []
    self.clicked_points = []

  def set_images(self):
    for filename in os.listdir(self.folder):
      img = cv2.imread(os.path.join(self.folder, filename))
      if img is None:
        print("!")
      if img is not None:
        self.images.append(img)
    self.clicked_points = [[] for i in range(len(self.images))]
  
  def get_images(self):
    return self.images
  
  def dump(self):
    for i, img in enumerate(self.images):
      cv2.imshow("image" + str(i + 1), img)
      cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  def make_image_point(self):
    for i, img in enumerate(self.images):
      cv2.imshow("image" + str(i + 1), img)
      cv2.waitKey(" ")
      cv2.destroyAllWindows()
  
  def click_event(self, event, x, y, flags, param): # マウスクリックイベントをハンドルする内部メソッド
    if event == cv2.EVENT_LBUTTONDOWN:
      self.clicked_points[param].append((x, y))
      print(f"Clicked at: ({x}, {y})")

  def show_images_and_get_clicks(self): # すべての画像を表示し、クリックされた位置を保存するメソッド
    for i, img in enumerate(self.images):
      cv2.imshow('Image', img)
      cv2.setMouseCallback('Image', self.click_event, i)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    
  def get_clicked_points(self, i):
    return np.array(self.clicked_points[i], dtype="double")

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
'''
image_points = np.array([
  (321, 257),
  (373, 250),
  (312, 249),
  (320, 230),
  (373, 224),
  (311, 222),
  (360, 218)
], dtype="double")
'''
in3
image_points = np.array([
  (358, 275),
  (407, 259),
  (322, 264),
  (357, 240),
  (407, 225),
  (321, 229),
  (371, 216),
], dtype="double")
'''

# Camera Intrinsic Paramter
dist_coeffs = np.array([
  [3.5503921110220771e-01,
  -1.8089641384590791e+00,
  -1.3224112624431249e-03,
  -1.1764534809794213e-03,
  2.7810334600309656e+00]
], dtype="double")
camera_matrix = np.array([
  [507.67984091855385, 0, 330.27513286880281],
  [0, 509.84029418345722, 242.96535901758909],
  [0, 0, 1]
], dtype="double")

origin = ObjectBox(469, 348, 238)
#origin = ObjectBox(217, 325, 138)
'''
pers = Perspactive()
pers.set_images()
pers.show_images_and_get_clicks()
print("Clicked Points")
print(pers.clicked_points)
'''

# これ以降はif __name__ == '__main__' ----------

print("\nSolve Perspective N Point Problem")

print("\nCamera Matrix")
print(camera_matrix)

print("\nDistortion Coefficient")
print(dist_coeffs)

print(origin.object_points())
#print(pers.get_clicked_points(0))
(success, rvec, tvec) = cv2.solvePnP(origin.object_points(), image_points, camera_matrix, dist_coeffs)
#(success, rvec, tvec) = cv2.solvePnP(origin.object_points(), pers.get_clicked_points(0), camera_matrix, dist_coeffs)
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

camera_direction = get_camera_direction(camera_matrix, tvec, rvec)
print("\nCamera Direction")
print(camera_direction)


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
plot_vector(plt, t_raw, camera_direction/100, color="green")
plt.show()

# ---------------
# ref: https://daily-tech.hatenablog.com/entry/2018/02/02/071655
#      https://python.atelierkobato.com/quiver
#      https://dev.classmethod.jp/articles/estimate-camera-external-parameter-matrix-2d-camera/
