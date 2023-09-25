import cv2
import numpy as np

# 画像ファイルのパスを指定
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# カメラのキャリブレーション行列と歪み係数を設定
fx = 1000  # X方向の焦点距離
fy = 1000  # Y方向の焦点距離
cx = 640   # 画像の中心X座標
cy = 360   # 画像の中心Y座標

camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

dist_coeffs = np.zeros((4, 1))

# 対応点の検出とマッチング
sift = cv2.SIFT_create()

# 画像と対応点を保存するリストを作成
images = []
keypoints_list = []
descriptors_list = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    images.append(image)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# 基準となる直方体の3D座標を定義
object_points = np.array([[0, 0, 0],  # 底面の中心
                          [1, 0, 0],  # 底面の右上
                          [1, 1, 0],  # 右上の隅
                          [0, 1, 0],  # 底面の左上
                          [0, 0, 1],  # 上面の中心
                          [1, 0, 1],  # 上面の右上
                          [1, 1, 1],  # 上面の右下
                          [0, 1, 1]]) # 上面の左下

# 三次元復元の実行
all_points_3d = []
all_points_2d = []

for i in range(len(image_paths)):
    for j in range(i + 1, len(image_paths)):
        # 対応点のマッチング
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_list[i], descriptors_list[j], k=2)

        # ランダムに対応点を選択
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 2D対応点を取得
        src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[j][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 三次元復元を実行
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, dst_pts, camera_matrix, dist_coeffs)
        points_3d = cv2.triangulatePoints(np.eye(3, 4), np.hstack((rvec, tvec)), src_pts, dst_pts)
        points_3d /= points_3d[3]  # ホモジニアス座標をカートesian座標に変換

        all_points_3d.append(points_3d)
        all_points_2d.append(dst_pts)

# 結果の可視化（各視点の3D点を合成して表示）
merged_points_3d = np.concatenate(all_points_3d, axis=1)
merged_points_2d = np.concatenate(all_points_2d, axis=0)

# 3D点の可視化
for i in range(merged_points_3d.shape[1]):
    x, y, z, _ = merged_points_3d[:, i]
    image_index = i // len(image_paths)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.circle(images[image_index], (int(merged_points_2d[i][0][0]), int(merged_points_2d[i][0][1])), 5, color, -1)
    cv2.putText(images[image_index], f"{x:.2f}, {y:.2f}, {z:.2f}", (int(merged_points_2d[i][0][0]), int(merged_points_2d[i][0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 画像の表示
for i, image in enumerate(images):
    cv2.imshow(f"Image {i+1}", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
