#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "urin_o_edge_img_sobel/common/include/common.hpp"

namespace urin_o_edge_img_sobel {

class UrinOEdgeImgSobelMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit UrinOEdgeImgSobelMPI(const InType &in);

 private:
  int height_ = 0;
  int width_ = 0;

  std::vector<int> input_pixels_;

  std::vector<int> local_pixels_;
  int local_height_ = 0;
  int local_height_with_halo_ = 0;

  void BroadcastParameters();
  void DistributeRows();
  std::vector<int> LocalGradientsComputing();
  int GradientX(int x, int y);
  int GradientY(int x, int y);
  void GatherResults(const std::vector<int> &local_result);

  void RowDistributionComputing(int world_rank, int world_size, int &base_rows, int &remainder, int &real_rows,
                                int &need_top_halo, int &need_bottom_halo, int &total_rows);

  void SendParameters(int world_rank, int world_size, int base_rows, int remainder,
                      std::vector<int> &real_rows_per_proc, std::vector<int> &send_counts,
                      std::vector<int> &send_displs) const;

  void DataDistribution(int world_rank, const std::vector<int> &send_counts, const std::vector<int> &send_displs);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace urin_o_edge_img_sobel
