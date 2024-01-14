import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import initialSetup as istu
import json

def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)

#

images = []
for filename in os.listdir(istu.image_dir):
    img = cv2.imread(os.path.join(istu.image_dir, filename))
    if img is None:
      print("画像の読み込みに失敗しました。")
    if img is not None:
      images.append(img)

clicked_points = [[] for _ in range(len(images))]
def click_event(event, x, y, flags, param): # マウスクリックイベントをハンドルする
  if event == cv2.EVENT_LBUTTONDOWN:
    clicked_points[param].append((x, y))
    print(f"Clicked at: ({x}, {y})")

def show_images_and_get_clicks(): # すべての画像を表示し、クリックされた位置を保存する
  for i, img in enumerate(images):
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event, i)
    print("\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
show_images_and_get_clicks()
print("\nclicked points")
print(clicked_points)

camera_poses = []
for i in range(len(images)):
  (_ , rvec, tvec) = cv2.solvePnP(istu.object_points, np.array(clicked_points[i], dtype="double"), istu.camera_matrix, istu.dist_coeffs)
  camera_poses.append((tvec, rvec))

print("\ncamera_poses")
def print_camera_poses():
  for rvec, tvec in camera_poses:
    rmat, _ = cv2.Rodrigues(rvec)
    rR = rmat
    tR = -rmat.T @ tvec
  print([tR, rR])
print_camera_poses()