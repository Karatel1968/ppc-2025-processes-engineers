#include "urin_o_edge_img_sobel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "urin_o_edge_img_sobel/common/include/common.hpp"

namespace urin_o_edge_img_sobel {

// Собельные ядра
constexpr std::array<std::array<int, 3>, 3> kSobelXArray = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};

constexpr std::array<std::array<int, 3>, 3> kSobelYArray = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

UrinOEdgeImgSobelMPI::UrinOEdgeImgSobelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
    input_pixels_ = std::get<0>(GetInput());
    height_ = std::get<1>(GetInput());
    width_ = std::get<2>(GetInput());
  }

  GetOutput().clear();
}

bool UrinOEdgeImgSobelMPI::ValidationImpl() {
  /*if (height_ <= 2 || width_ <= 2) return false;
  if (static_cast<int>(input_pixels_.size()) != height_ * width_) return false;*/
  return true;
}

bool UrinOEdgeImgSobelMPI::PreProcessingImpl() {
  // Просто резервируем место для выходных данных (полностью)
  /*GetOutput().resize(height_ * width_, 0);
  return true;*/
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (height_ <= 2 || width_ <= 2) {
      return false;
    }
    if (static_cast<int>(input_pixels_.size()) != height_ * width_) {
      return false;
    }
  }

  return true;
}

void UrinOEdgeImgSobelMPI::BroadcastParameters() {
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void UrinOEdgeImgSobelMPI::RowDistributionComputing(int world_rank, int world_size, int &base_rows, int &remainder,
                                                    int &real_rows, int &need_top_halo, int &need_bottom_halo,
                                                    int &total_rows) {
  base_rows = height_ / world_size;
  remainder = height_ % world_size;

  real_rows = base_rows + (world_rank < remainder ? 1 : 0);
  local_height_ = real_rows;

  need_top_halo = (world_rank > 0) ? 1 : 0;
  need_bottom_halo = (world_rank < (world_size - 1)) ? 1 : 0;

  total_rows = real_rows + need_top_halo + need_bottom_halo;
  local_height_with_halo_ = total_rows;

  local_pixels_.resize(static_cast<size_t>(total_rows) * width_, 0);
}

void UrinOEdgeImgSobelMPI::SendParameters(int world_rank, int world_size, int base_rows, int remainder,
                                          std::vector<int> &real_rows_per_proc, std::vector<int> &send_counts,
                                          std::vector<int> &send_displs) const {
  /*if (world_rank == 0) {
    int current_row = 0;
    for (int dest = 0; dest < world_size; ++dest) {
      int dest_real_rows = base_rows + (dest < remainder ? 1 : 0);
      real_rows_per_proc[dest] = dest_real_rows;

      int dest_need_top_halo = (dest > 0) ? 1 : 0;
      int dest_need_bottom_halo = (dest < (world_size - 1)) ? 1 : 0;

      int start_row_with_halo = current_row - dest_need_top_halo;
      if (start_row_with_halo < 0) {
        start_row_with_halo = 0;
      }

      int end_row_with_halo = current_row + dest_real_rows + dest_need_bottom_halo - 1;
      if (end_row_with_halo >= height_) {
        end_row_with_halo = height_ - 1;
      }

      int actual_rows = end_row_with_halo - start_row_with_halo + 1;

      send_counts[dest] = actual_rows * width_;
      send_displs[dest] = start_row_with_halo * width_;

      current_row += dest_real_rows;
    }
  }*/
  if (world_rank != 0) {
    return;
  }

  int current_row = 0;
  for (int dest = 0; dest < world_size; ++dest) {
    const bool dest_has_extra = dest < remainder;
    const int dest_real_rows = base_rows + (dest_has_extra ? 1 : 0);
    real_rows_per_proc[dest] = dest_real_rows;

    const bool needs_top_halo = dest > 0;
    const bool needs_bottom_halo = dest < (world_size - 1);

    int start_row_with_halo = current_row - (needs_top_halo ? 1 : 0);
    start_row_with_halo = std::max(start_row_with_halo, 0);

    int end_row_with_halo = current_row + dest_real_rows + (needs_bottom_halo ? 1 : 0) - 1;
    end_row_with_halo = std::min(end_row_with_halo, height_ - 1);

    const int actual_rows = end_row_with_halo - start_row_with_halo + 1;

    send_counts[dest] = actual_rows * width_;
    send_displs[dest] = start_row_with_halo * width_;

    current_row += dest_real_rows;
  }
}

void UrinOEdgeImgSobelMPI::DataDistribution(int world_rank, const std::vector<int> &send_counts,
                                            const std::vector<int> &send_displs) {
  MPI_Scatterv(world_rank == 0 ? input_pixels_.data() : nullptr, send_counts.data(), send_displs.data(), MPI_INT,
               local_pixels_.data(), static_cast<int>(local_pixels_.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void UrinOEdgeImgSobelMPI::DistributeRows() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int base_rows = 0;
  int remainder = 0;
  int real_rows = 0;
  int need_top_halo = 0;
  int need_bottom_halo = 0;
  int total_rows = 0;

  RowDistributionComputing(world_rank, world_size, base_rows, remainder, real_rows, need_top_halo, need_bottom_halo,
                           total_rows);

  std::vector<int> send_counts(world_size, 0);
  std::vector<int> send_displs(world_size, 0);
  std::vector<int> real_rows_per_proc(world_size, 0);

  SendParameters(world_rank, world_size, base_rows, remainder, real_rows_per_proc, send_counts, send_displs);

  DataDistribution(world_rank, send_counts, send_displs);
}

int UrinOEdgeImgSobelMPI::GradientX(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[(static_cast<size_t>(ny) * width_) + nx];
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

int UrinOEdgeImgSobelMPI::GradientY(int x, int y) {
  int sum = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int nx = x + kx;
      int ny = y + ky;

      if (nx >= 0 && nx < width_ && ny >= 0 && ny < local_height_with_halo_) {
        int pixel = local_pixels_[(static_cast<size_t>(ny) * width_) + nx];
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

std::vector<int> UrinOEdgeImgSobelMPI::LocalGradientsComputing() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> local_result;
  if (local_height_ > 0) {
    local_result.resize(static_cast<size_t>(local_height_) * width_, 0);

    for (int local_y = 0; local_y < local_height_; ++local_y) {
      int y = local_y + ((world_rank > 0) ? 1 : 0);

      for (int xx = 0; xx < width_; ++xx) {
        int gx = GradientX(xx, y);
        int gy = GradientY(xx, y);
        int mag = static_cast<int>(std::sqrt((gx * gx) + (gy * gy)));
        local_result[(static_cast<size_t>(local_y) * width_) + xx] = std::min(mag, 255);
      }
    }
  }

  return local_result;
}

void UrinOEdgeImgSobelMPI::GatherResults(const std::vector<int> &local_result) {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> local_result_sizes(world_size);
  int local_size = static_cast<int>(local_result.size());
  MPI_Allgather(&local_size, 1, MPI_INT, local_result_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(world_size);
  int total_size = 0;
  for (int i = 0; i < world_size; ++i) {
    displs[i] = total_size;
    total_size += local_result_sizes[i];
  }

  GetOutput().resize(static_cast<size_t>(total_size));

  MPI_Allgatherv(local_result.empty() ? nullptr : local_result.data(), local_size, MPI_INT, GetOutput().data(),
                 local_result_sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
}

bool UrinOEdgeImgSobelMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastParameters();
  DistributeRows();

  auto local_result = LocalGradientsComputing();

  GatherResults(local_result);

  return true;
}

bool UrinOEdgeImgSobelMPI::PostProcessingImpl() {
  return true;
}

}  // namespace urin_o_edge_img_sobel
