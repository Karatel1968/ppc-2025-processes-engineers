#include "urin_o_edge_img_sobel/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "urin_o_edge_img_sobel/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_edge_img_sobel {

// Собельные ядра
const int kSobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const int kSobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

UrinOEdgeImgSobelSEQ::UrinOEdgeImgSobelSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  input_pixels_ = std::get<0>(GetInput());
  height_ = std::get<1>(GetInput());
  width_ = std::get<2>(GetInput());

  GetOutput().resize(height_ * width_, 0);
}

bool UrinOEdgeImgSobelSEQ::ValidationImpl() {
  if (height_ <= 2 || width_ <= 2) {
    return false;
  }
  if (static_cast<int>(input_pixels_.size()) != height_ * width_) {
    return false;
  }
  return true;
}

bool UrinOEdgeImgSobelSEQ::PreProcessingImpl() {
  return true;
}

int UrinOEdgeImgSobelSEQ::GradientX(int x, int y) {
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        int pixel = input_pixels_[static_cast<size_t>(ny) * width_ + nx];
        sum += pixel * kSobelX[ky + 1][kx + 1];
      }
    }
  }
  return sum;
}

int UrinOEdgeImgSobelSEQ::GradientY(int x, int y) {
  int sum = 0;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        int pixel = input_pixels_[static_cast<size_t>(ny) * width_ + nx];
        sum += pixel * kSobelY[ky + 1][kx + 1];
      }
    }
  }
  return sum;
}

bool UrinOEdgeImgSobelSEQ::RunImpl() {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int gx = GradientX(x, y);
      int gy = GradientY(x, y);
      int mag = static_cast<int>(std::sqrt(gx * gx + gy * gy));
      GetOutput()[static_cast<size_t>(y) * width_ + x] = std::min(mag, 255);
    }
  }
  return true;
}

bool UrinOEdgeImgSobelSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace urin_o_edge_img_sobel
