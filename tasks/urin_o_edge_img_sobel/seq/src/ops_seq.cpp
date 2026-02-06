#include "urin_o_edge_img_sobel/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "urin_o_edge_img_sobel/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace urin_o_edge_img_sobel {

// Собельные ядра
constexpr std::array<std::array<int, 3>, 3> kSobelXArray = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};

constexpr std::array<std::array<int, 3>, 3> kSobelYArray = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

UrinOEdgeImgSobelSEQ::UrinOEdgeImgSobelSEQ(const InType &in)
    : input_pixels_(std::get<0>(in)), height_(std::get<1>(in)), width_(std::get<2>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  /*input_pixels_ = std::get<0>(GetInput());
  height_ = std::get<1>(GetInput());
  width_ = std::get<2>(GetInput());*/
  const std::size_t total_size = static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_);

  GetOutput().resize(total_size, 0);
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
        int pixel = input_pixels_[(static_cast<size_t>(ny) * width_) + nx];
        // const int kernel_value = kSobelX[static_cast<size_t>(ky + 1)][static_cast<size_t>(kx + 1)];
        const int sobel_x_idx = kx + 1;
        const int sobel_y_idx = ky + 1;
        const int kernel_value = kSobelXArray.at(static_cast<size_t>(sobel_y_idx)).at(static_cast<size_t>(sobel_x_idx));
        sum += pixel * kernel_value;
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
        int pixel = input_pixels_[(static_cast<size_t>(ny) * width_) + nx];
        // const int kernel_value = kSobelY[static_cast<size_t>(ky + 1)][static_cast<size_t>(kx + 1)];
        const int sobel_x_idx = kx + 1;
        const int sobel_y_idx = ky + 1;

        // Преобразуем только при вызове .at()
        const int kernel_value = kSobelYArray.at(static_cast<size_t>(sobel_y_idx)).at(static_cast<size_t>(sobel_x_idx));
        sum += pixel * kernel_value;
      }
    }
  }
  return sum;
}

bool UrinOEdgeImgSobelSEQ::RunImpl() {
  for (int ky = 0; ky < height_; ++ky) {
    for (int kx = 0; kx < width_; ++kx) {
      int gx = GradientX(kx, ky);
      int gy = GradientY(kx, ky);
      int mag = static_cast<int>(std::sqrt((gx * gx) + (gy * gy)));
      GetOutput()[(static_cast<size_t>(ky) * width_) + kx] = std::min(mag, 255);
    }
  }
  return true;
}

bool UrinOEdgeImgSobelSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace urin_o_edge_img_sobel
