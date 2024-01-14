import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import initialSetup as istu
import json

'''
BASE_DIR = "/Users/keisaiki/Documents/Lab/3DReconstruction/reconstruction/projects/"
project_name = input("プロジェクト名を入力してください。\n")
input_json_file_path = BASE_DIR + project_name + "/json/istu.json"
with open(input_json_file_path, "r") as file:
  data = json.load(file)
image_dir = data["image_dir"]
print("\nimage_dir")
print(image_dir)
'''
  

def plot_vector(axes, location, vector, color="red"):
  axes.quiver(location[0], location[1], location[2],
              vector[0], vector[1], vector[2],
              color=color, length=1,
              arrow_length_ratio=0.1)

# 

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

#

images = []
for filename in os.listdir(istu.image_dir):
    img = cv2.imread(os.path.join(istu.image_dir, filename))
    if img is None:
      print("画像の読み込みに失敗しました。")
    if img is not None:
      images.append(img)

clicked_points = [[] for _ in range(len(images))]
show_images_and_get_clicks()
print("clicked points")
print(clicked_points)

camera_poses = []
for i in range(len(images)):
  (_ , tvec, rvec) = cv2.solvePnP(istu.object_points, np.array(clicked_points[i], dtype="double"), istu.camera_matrix, istu.dist_coeffs)
  camera_poses.append((tvec, rvec))

print("camera_poses")
print(camera_poses)