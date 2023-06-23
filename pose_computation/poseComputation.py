import numpy as np
import cv2
import math

world = np.array([
  (0., 0., 0.),
  (222., 0., 0.),
  (0., 116., 0.),
  (0., 0., 60.),
  (222., 0., 60.),
  (222., 116., 60.),
  (0., 116., 60.),
])

img_pnts = np.array([
  (310, 303),
  (436, 246),
  (244, 268),
  (311, 248),
  (441, 202),
  (372, 183),
  (241, 219),
], dtype = "double")

# Camera Intrinsic Paramter
dist_coeffs = np.array([
  [3.63211720e-01,
   -1.93851026e+00,
   -1.47109243e-04,
   5.95798535e-04,
   3.14452169e+00]
])

camera_matrix = np.array(
						[[508.98721069, 0., 331.12232423],
            [0., 511.00672888, 243.52320381],
            [0., 0., 1.]], dtype = "double"
						)

if __name__ == "__main__":
  print("Solve Perspective N Point Problem")

  print("\nCamera Matrix : ")
  print(camera_matrix)

  print("\nDistortion Coefficient")
  print(dist_coeffs)
  
  (success, rot_vec, trans_vec) = cv2.solvePnP(world, img_pnts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
  
  print("\nTranslation Vector")
  print(trans_vec)
  
  print("\nRotation Vector")
  print(rot_vec)
  
  print("\nRotation Matrix")
  R, jacob = cv2.Rodrigues(rot_vec)
  print(R)
  
# ref: https://daily-tech.hatenablog.com/entry/2018/02/02/071655
  