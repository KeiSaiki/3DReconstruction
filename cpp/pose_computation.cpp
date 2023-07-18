#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

const std::string input_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/info";

int main() {
  cv::FileStorage fs(input_filepath + "/cam_mat.xml", cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "File cannot be opened" << std::endl;
    exit(1);
  }

  cv::Mat cam_mat;
  cv::Mat dist_coeffs;
  auto cam_mat_filenode = fs["intrinsic"];
  auto dist_coeffs_filenode = fs["distortion"];
  cam_mat_filenode >> cam_mat;
  dist_coeffs_filenode >> dist_coeffs;
  std::vector<cv::Point3f> obj_points = {
    {0, 0, 0},
    {222, 0, 0},
    {0, 116, 0},
    {0, 0, 60},
    {222, 0, 60},
    {222, 116, 60},
    {0, 116, 60}
  };
  std::vector<cv::Point2f> img_points = {
    {310, 303},
    {436, 246},
    {244, 268},
    {311, 248},
    {441, 202},
    {372, 183},
    {241, 219}
  };
  cv::Mat rvec;
  cv::Mat tvec;
  
  cv::solvePnP(obj_points, img_points, cam_mat, dist_coeffs, rvec, tvec);
  std::cout << rvec << std::endl;
  std::cout << tvec << std::endl;
}

//g++ -std=c++17 pose_computation.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
