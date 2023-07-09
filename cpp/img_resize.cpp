#include <opencv2/opencv.hpp>
#include <string>

std::string input_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/calibration_img_original";
std::string output_filepath = "/Users/keisaiki/Documents/Lab/3DReconstruction/cpp/calibration_img_resized";

int main() {
  int output_img_num = 0;
  for (int i = 0; i < 10000; i++) {
    std::string input_img_num_str = std::to_string(i);
    input_img_num_str = std::string(4 - static_cast<int>(input_img_num_str.size()), '0') + input_img_num_str; //0埋め
    cv::Mat img = cv::imread(input_filepath + "/IMG_" + input_img_num_str + ".jpeg");
    if (img.empty()) continue;

    cv::resize(img, img, cv::Size(), img.cols/6, img.rows/6);
    std::string output_img_num_str = std::to_string(output_img_num++);
    output_img_num_str = std::string(3 - static_cast<int>(output_img_num_str.size()), '0') + output_img_num_str; //0埋め
    cv::imwrite(output_filepath + "/img_" + output_img_num_str + ".jpeg", img);
  }
}

//g++ -std=c++11 img_resize.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`
