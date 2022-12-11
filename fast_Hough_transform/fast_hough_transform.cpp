#include <algorithm>
#include <cmath>
#include "fast_hough_transform.h"

#define M_PI 3.14159265358979323846

static int prod_n_sqrt_2(int n) { return int(n * sqrt(2)); }

FastHoughTransformer::FastHoughTransformer(const cv::Mat& image_matrix) {
  const int kPiDegree = 180;
  int width = image_matrix.cols;
  int height = image_matrix.rows;

  this->N = std::pow(
      2,
      ceil(log2(double(std::max(width, height)))));  // следующая степень двойки
  this->markInDegree = 2 * this->N / kPiDegree;
  cv::Mat img_with_padding(N, N, image_matrix.type());

  cv::copyMakeBorder(image_matrix, img_with_padding, 0, N - height, 0,
                     N - width, cv::BORDER_CONSTANT, 0);

  cv::Mat new_drt((kPiDegree / 2) * this->markInDegree,
                 prod_n_sqrt_2(this->N) / 2 + prod_n_sqrt_2(this->N),
                 CV_32S, cv::Scalar(0));
  cv::swap(this->new_image_matrix, new_drt);

  // Разделить на каналы
  std::vector<cv::Mat> channels;
  cv::split(img_with_padding, channels);

  // Построить таблицу преобразования Радона
  for (cv::Mat& channel : channels) {
    radon_transform(channel);
  }

  double minim;
  double maxim;
  cv::Point min_loc;
  cv::Point max_loc;
  cv::minMaxLoc(this->new_image_matrix, &minim, &maxim, &min_loc, &max_loc);

  int cur_width = this->new_image_matrix.cols;
  int cur_height = this->new_image_matrix.rows;

  for (int row = 0; row < cur_height; ++row) {
    for (int col = 0; col < cur_width; ++col) {
      int val = int((double)(this->new_image_matrix.at<int>(row, col) - minim) /
                    (maxim - minim) * 255.);
      this->new_image_matrix.at<int>(row, col) = val;
    }
  }
}
/////////////////////////////////////////////////////////////////

void FastHoughTransformer::radon_transform(const cv::Mat& image_matrix) {
  cv::Mat image_matrix_transpose = image_matrix.t();
  cv::Mat to_clock(N, N, image_matrix.type());
  cv::rotate(image_matrix, to_clock, cv::ROTATE_90_CLOCKWISE);
  cv::Mat IclockwiseT = to_clock.t();

  cv::Mat R_clock = tranform_from_zero_to_pi_div_4(to_clock);

  for (int row = 0; row < this->new_image_matrix.rows / 2; ++row) {
    for (int col = -prod_n_sqrt_2(this->N) / 2; col < N; ++col) {
      double thetaRad = (double)row * (M_PI / 2) / this->new_image_matrix.rows;
      int x = int(round(abs(col) / cos(thetaRad)));
      if (N <= x) {
        continue;
      }
      if (col < 0) {
        x = 2 * N - x;
      }
      int y = int(round(N * tan(thetaRad)));
      this->new_image_matrix.at<int>(this->new_image_matrix.rows / 2 - 1 - row,
                         prod_n_sqrt_2(this->N) / 2 + col) +=
          R_clock.at<int>(y, x);
    }
  }
  cv::Mat Rt = tranform_from_zero_to_pi_div_4(image_matrix_transpose);

  for (int row = 0; row < this->new_image_matrix.rows / 2; ++row) {
    for (int col = 0; col < prod_n_sqrt_2(this->N); ++col) {
      double thetaRad = (double)row * (M_PI / 2) / this->new_image_matrix.rows;
      int y = int(round(N * tan(thetaRad)));

      int x = N - int(round(col / cos(thetaRad)));
      if (x < 0) {
        if (y + x < 0) {
          continue;
        }
        x = 2 * N + x;
      }

      this->new_image_matrix.at<int>(this->new_image_matrix.rows / 2 + row,
                         prod_n_sqrt_2(this->N) / 2 + col) +=
          Rt.at<int>(y, x);
    }
  }
}

cv::Mat FastHoughTransformer::tranform_from_zero_to_pi_div_4(
    const cv::Mat& image_matrix) {
  cv::Mat R(N, 2 * N, CV_32S, cv::Scalar(0));
  image_matrix.copyTo(R(cv::Rect(0, 0, N, N)));

  cv::Mat new_R(N, 2 * N, CV_32S, cv::Scalar(0));

  for (int i = 1; i <= log2(N); ++i) {
    int delta_y = pow(2, i);
    int prev_delta_y = pow(2, i - 1);

    for (int y = 0; y < N; y += delta_y) {
      for (int x = 0; x < 2 * N; ++x) {
        for (int a = 0; a < delta_y; ++a) {
          int upStick = R.at<int>(y + a / 2, x);
          int down_stick = R.at<int>(y + prev_delta_y + a / 2,
                                    (x + (a + 1) / 2) % (2 * N));

          new_R.at<int>(y + a, x) = upStick + down_stick;
        }
      }
    }
    cv::swap(R, new_R);
  }
  return R;
}

int FastHoughTransformer::apply() {
  const int kPiDegree = 180;
  double max_dispersion = 0.;
  int slant_angle = 0;

  int delta_angle = 1;

  for (int angle = -kPiDegree / 4; angle < kPiDegree / 4; angle += delta_angle) {
    uint64_t sum = 0;
    uint64_t sumSquared = 0;
    int count = 0;

    for (int row = 0; row < this->markInDegree * delta_angle; ++row) {
      for (int col = 0; col < this->new_image_matrix.cols; ++col) {
        uint64_t val = (uint64_t)this->new_image_matrix.at<int>(
            (angle + kPiDegree / 4) * this->markInDegree + row, col);
        count += val == 0 ? 0 : 1;
        sum += val;
        sumSquared += val * val;
      }
    }

    if (count == 0) {
      continue;
    }

    double mean = (double)sum / count;
    double dispersion = (double)sumSquared / count - mean * mean;

    if (max_dispersion < dispersion) {
      max_dispersion = dispersion;
      slant_angle = angle;
    }
  }

  return slant_angle;
}

void FastHoughTransformer::imReadAfterTransformation(std::string fileName) {
  cv::imwrite(fileName, this->new_image_matrix);
}
