#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include "fast_hough_transform.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "you should type input_image_path, output_image_path" << '\n';
    return 0;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  cv::Mat src = cv::imread(input);
  cv::Mat gray_src(src);
  cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

  auto begin = std::chrono::system_clock::now();

  FastHoughTransformer fht_Scheduler(gray_src);
  int slant_angle = fht_Scheduler.apply();

  auto end = std::chrono::system_clock::now();

  std::cout<<"Slant angle = "<<slant_angle <<'\n';
  fht_Scheduler.imReadAfterTransformation(output);

  auto diff =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  std::cout<<"Fast Hought transformation time = "<< diff.count() << " milliseconds"<<"/"<<"n"<<'\n';

  return 0;
}
