import os
import shutil
import numpy as np
import json

project_name = input("プロジェクト名を入力してください。\n")
BASE_DIR = "/Users/keisaiki/Documents/Lab/3DReconstruction/reconstruction/projects/"
project_dir = BASE_DIR + project_name + "/"
original_image_dir = project_dir + "images/"
image_dir = project_dir + "images_resized/"
json_file_path = project_dir + "json/istu.json"

def make_project_directory():
    if os.path.exists(project_dir):
        response = input(f"プロジェクト{project_name}は既に存在します。中身を全て削除して上書きしますか？ [y/n]: ").strip().lower()
        if response == 'y':
            shutil.rmtree(project_dir)
            os.makedirs(original_image_dir)
            os.makedirs(image_dir)
            print(f"プロジェクト{project_name}が上書きされました。")
        else:
            print("操作はキャンセルされました。")
    else:
        os.makedirs(original_image_dir)
        os.makedirs(image_dir)
        print(f"プロジェクト{project_name}を作成しました。")

make_project_directory()

# カメラ行列の設定 
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

# マーカーの設定 

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

object_points = ObjectBox(469, 348, 238).object_points()

# json書き出し
data = [
  {"project_dir": project_dir,
  "image_dir": image_dir,
  "camera_matrix": camera_matrix.tolist(),
  "dist_coeffs": dist_coeffs.tolist(),
  "object_point": object_points.tolist()}
]

'''
with open(json_file_path, "w") as file:
    json.dump({}, file)
with open(json_file_path, "w") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
'''