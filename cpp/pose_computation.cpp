#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

const std::string input_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/info";

const double PI = 3.141592653589793;

class ObjectBox {
  public:
    ObjectBox(float x, float y, float z) : _x(x), _y(y), _z(z) {}

    std::vector<cv::Point3f> object_points () {
      std::vector<cv::Point3f> result = {
        {0, 0, 0},
        {_x, 0, 0},
        {0, _y ,0},
        {0, 0, _z},
        {_x, 0, _z},
        {_x, _y, _z},
        {0, _y, _z}
      };
      return result;
    }

  private:
    const float _x, _y, _z;
};

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
  ObjectBox obj_box(469, 348, 238);
  std::vector<cv::Point3f> obj_points = obj_box.object_points();
  std::vector<cv::Point2f> img_points = {
    {302, 365},
    {367, 319},
    {212, 342},
    {303, 287},
    {370, 259},
    {292, 247},
    {211, 271}
  };
  cv::Mat rvec;
  cv::Mat tvec;
  
  cv::solvePnP(obj_points, img_points, cam_mat, dist_coeffs, rvec, tvec);
  std::cout << rvec << std::endl;
  std::cout << tvec << std::endl;
  cv::Mat rmat;
  Rodrigues(rvec, rmat);
  std::cout << rmat/PI*180 << std::endl;
  //std::cout << tvec.rows << " " << tvec.cols << " " << tvec.channels() <<  std::endl;
  /*
  */
  double len = 0;
  for (int i = 0; i < 3; i++) len += tvec.at<double>(i, 0)*tvec.at<double>(i, 0);
  len = sqrt(len);
  std::cout << len << std::endl;
  std::cout << rmat*tvec << std::endl;
}

//g++ -std=c++17 pose_computation.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
//三角形の下の点からカメラまで約2400mm