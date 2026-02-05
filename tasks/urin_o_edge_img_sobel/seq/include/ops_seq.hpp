#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "urin_o_edge_img_sobel/common/include/common.hpp"

namespace urin_o_edge_img_sobel {

class UrinOEdgeImgSobelSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit UrinOEdgeImgSobelSEQ(const InType &in);

 private:
  std::vector<int> input_pixels_;
  int height_ = 0;
  int width_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int GradientX(int x, int y);
  int GradientY(int x, int y);
};

}  // namespace urin_o_edge_img_sobel
