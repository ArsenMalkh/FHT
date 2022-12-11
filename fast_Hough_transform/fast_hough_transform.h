#pragma once
#include <opencv2/opencv.hpp>

class FastHoughTransformer {
 public:
  FastHoughTransformer(const cv::Mat& image_matrix);
  int apply();

  void imReadAfterTransformation(std::string fileName);

 private:
  int markInDegree;
  int N;
  cv::Mat new_image_matrix;

 private:
  void radon_transform(const cv::Mat& image_matrix);
  cv::Mat tranform_from_zero_to_pi_div_4(const cv::Mat& image_matrix);
};
