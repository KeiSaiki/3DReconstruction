#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

const std::string input_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/calibration_img_resized";
const std::string output_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/info";

struct pattern {
  int num_of_img;
  int points;
  int row = 7;
  int col = 10; 
  int size = row*col;
  cv::Size size_cv = cv::Size2i(col, row);
  double chess_size = 23.3;
  pattern(const int& num) {
    num_of_img = num;
    points = num_of_img*size;
  } 
};

int main() {
  std::vector<cv::Mat> src_imgs;
  int count = 0;
  for (int i = 0; i < 1000; i++) {
    std::string input_str = std::to_string(i);
    input_str = std::string(3 - static_cast<int>(input_str.size()), '0') + input_str;
    cv::Mat img = cv::imread(input_filepath + "/img_" + input_str + ".jpeg");
    if (img.empty()) {
      std::cerr << "cannnot load img_" + input_str + ".jpeg" << std::endl;
      continue;
    }
    src_imgs.push_back(img);
    count++;
  }
  pattern pat(count);

  std::vector<cv::Point3f> object;
  for (int i = 0; i < pat.row; i++) {
    for (int j = 0; j < pat.col; j++) {
      object.emplace_back(pat.chess_size*i, pat.chess_size*j, 0.);
    }
  }
  std::vector<std::vector<cv::Point3f>> object_points;
  for (int i = 0; i < pat.num_of_img; i++) {
    object_points.push_back(object);
  }

  int found_num = 0;
  std::vector<cv::Point2f> corners;
  std::vector<std::vector<cv::Point2f>> img_points;
  cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);
  for (int i = 0; i < pat.num_of_img; i++) {
    auto found = cv::findChessboardCorners(src_imgs[i], pat.size_cv, corners);
    if (found) {
      std::cout << "img_" << std::setfill('0') << std::setw(3) << i << " ...succeeded" << std::endl;
      found_num++;
    } else {
      std::cerr << "img_" << std::setfill('0') << std::setw(3) << i << " ...failed" << std::endl;
    }

    cv::Mat src_img_gray = cv::Mat(src_imgs[i].size(), CV_8UC1);
    cv::cvtColor(src_imgs[i], src_img_gray, cv::COLOR_BGR2GRAY);
    cv::find4QuadCornerSubpix(src_img_gray, corners, cv::Size(3, 3));
    cv::drawChessboardCorners(src_imgs[i], pat.size_cv, corners, found);
    img_points.push_back(corners);

    cv::imshow("Calibration", src_imgs[i]);
    cv::waitKey(0);
  }
  cv::destroyWindow("Calibration");

  if (found_num != pat.num_of_img) {
    std::cerr << "Calibration Images are insufficient." << std::endl;
    std::exit(1);
  }

  cv::Mat cam_mat;
  cv::Mat dist_coefs;
  std::vector<cv::Mat> rvecs, tvecs;
  cv::calibrateCamera(object_points, img_points, src_imgs[0].size(), cam_mat, dist_coefs, rvecs, tvecs);

  cv::FileStorage fs(output_filepath + "/cam_mat.xml", cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "File can not be opend." << std::endl;
    std::exit(1);
  }
  fs << "intrinsic" << cam_mat;
  fs << "distortion" << dist_coefs;
  fs.release();

  return 0;
}

//g++ -std=c++11 calibration.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
