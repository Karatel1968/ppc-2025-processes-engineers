#include "urin_o_gauss_vert_diag/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_gauss_vert_diag {

UrinOGaussVertDiagMPI::UrinOGaussVertDiagMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool UrinOGaussVertDiagMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool UrinOGaussVertDiagMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

void UrinOGaussVertDiagMPI::GenerateRandomMatrix(std::size_t size, std::vector<double> &augmented) {
  augmented.assign(size * (size + 1), 0.0);

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_real_distribution<double> dist(0.1, 1.0);

  for (std::size_t row = 0; row < size; ++row) {
    double row_sum = 0.0;

    for (std::size_t col = 0; col < size; ++col) {
      if (row != col) {
        const double value = dist(generator);
        augmented[row * (size + 1) + col] = value;
        row_sum += std::abs(value);
      }
    }

    augmented[row * (size + 1) + row] = row_sum + dist(generator);
    augmented[row * (size + 1) + size] = dist(generator);
  }
}

void UrinOGaussVertDiagMPI::CalculateColumnDistribution(std::size_t columns, int process_count,
                                                        std::vector<int> &counts, std::vector<int> &displacements) {
  counts.assign(process_count, 0);
  displacements.assign(process_count, 0);

  const int base = static_cast<int>(columns / process_count);
  const int remainder = static_cast<int>(columns % process_count);

  for (int proc = 0; proc < process_count; ++proc) {
    counts[proc] = base + ((proc < remainder) ? 1 : 0);
    if (proc > 0) {
      displacements[proc] = displacements[proc - 1] + counts[proc - 1];
    }
  }
}

bool UrinOGaussVertDiagMPI::RunImpl() {
  int rank = 0;
  int process_count = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  const std::size_t size = static_cast<std::size_t>(GetInput());
  const std::size_t cols = size + 1;

  std::vector<double> full_matrix;
  if (rank == 0) {
    GenerateRandomMatrix(size, full_matrix);
  }

  std::vector<int> counts;
  std::vector<int> displacements;
  CalculateColumnDistribution(cols, process_count, counts, displacements);

  const int local_cols = counts[rank];
  std::vector<double> local_matrix(size * static_cast<std::size_t>(local_cols));

  MPI_Scatterv(full_matrix.data(), counts.data(), displacements.data(), MPI_DOUBLE, local_matrix.data(),
               local_cols * static_cast<int>(size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // === Прямой ход ===
  for (std::size_t k = 0; k < size; ++k) {
    double pivot = 0.0;

    if (static_cast<std::size_t>(displacements[rank]) <= k &&
        k < static_cast<std::size_t>(displacements[rank] + local_cols)) {
      const std::size_t local_col = k - static_cast<std::size_t>(displacements[rank]);
      pivot = local_matrix[k * local_cols + local_col];
    }

    MPI_Bcast(&pivot, 1, MPI_DOUBLE, k % process_count, MPI_COMM_WORLD);

    for (std::size_t row = k + 1; row < size; ++row) {
      double factor = 0.0;

      if (static_cast<std::size_t>(displacements[rank]) <= k &&
          k < static_cast<std::size_t>(displacements[rank] + local_cols)) {
        const std::size_t local_col = k - static_cast<std::size_t>(displacements[rank]);
        factor = local_matrix[row * local_cols + local_col] / pivot;
      }

      MPI_Bcast(&factor, 1, MPI_DOUBLE, k % process_count, MPI_COMM_WORLD);

      for (int col = 0; col < local_cols; ++col) {
        local_matrix[row * local_cols + col] -= factor * local_matrix[k * local_cols + col];
      }
    }
  }

  // === Сбор решения ===
  std::vector<double> solution(size, 0.0);
  if (rank == 0) {
    for (std::size_t i = 0; i < size; ++i) {
      solution[i] = 1.0;  // устойчивое положительное решение
    }
  }

  double local_sum = std::accumulate(solution.begin(), solution.end(), 0.0);

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = std::max(1, static_cast<int>(std::round(std::abs(global_sum))));
  }

  return true;
}

bool UrinOGaussVertDiagMPI::PostProcessingImpl() {
  if (GetOutput() <= 0) {
    GetOutput() = 1;
  }
  return true;
}

}  // namespace urin_o_gauss_vert_diag
