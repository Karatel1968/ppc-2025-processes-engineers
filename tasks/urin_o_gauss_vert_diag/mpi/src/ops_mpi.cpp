#include "urin_o_gauss_vert_diag/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
// #include "util/include/util.hpp"

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

  std::mt19937 gen(123);
  std::uniform_real_distribution<double> off_diag(0.1, 1.0);
  std::uniform_real_distribution<double> diag_add(1.0, 5.0);
  std::uniform_real_distribution<double> rhs_dist(1.0, 10.0);

  for (std::size_t row = 0; row < size; ++row) {
    double sum = 0.0;
    for (std::size_t col = 0; col < size; ++col) {
      if (row != col) {
        const double v = off_diag(gen);
        augmented[(row * (size + 1)) + col] = v;
        sum += std::abs(v);
      }
    }
    augmented[(row * (size + 1)) + row] = sum + diag_add(gen);
    augmented[(row * (size + 1)) + size] = rhs_dist(gen);
  }
}

int UrinOGaussVertDiagMPI::FindOwner(std::size_t global_row, const std::vector<int> &displs,
                                     const std::vector<int> &rows_per_proc) {
  for (std::size_t i = 0; i < displs.size(); ++i) {
    const std::size_t begin = static_cast<std::size_t>(displs[i]);
    const std::size_t end = begin + static_cast<std::size_t>(rows_per_proc[i]);
    if (global_row >= begin && global_row < end) {
      return static_cast<int>(i);
    }
  }
  return 0;
}

bool UrinOGaussVertDiagMPI::RunImpl() {
  int rank = 0;
  int proc_count = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  const auto size = static_cast<std::size_t>(GetInput());
  const std::size_t row_width = size + 1;

  // -------- Распределение строк --------
  std::vector<int> rows_per_proc(proc_count, 0);
  std::vector<int> displs(proc_count, 0);

  for (int i = 0; i < proc_count; ++i) {
    rows_per_proc[i] = static_cast<int>(size / proc_count);
    if (static_cast<std::size_t>(i) < size % proc_count) {
      rows_per_proc[i]++;
    }
  }

  std::partial_sum(rows_per_proc.begin(), rows_per_proc.end() - 1, displs.begin() + 1);

  const std::size_t local_rows = static_cast<std::size_t>(rows_per_proc[rank]);

  std::vector<double> local_matrix(local_rows * row_width);
  std::vector<double> full_matrix;

  if (rank == 0) {
    GenerateRandomMatrix(size, full_matrix);
  }

  std::vector<int> send_counts(proc_count);
  std::vector<int> send_displs(proc_count);

  for (int i = 0; i < proc_count; ++i) {
    send_counts[i] = rows_per_proc[i] * static_cast<int>(row_width);
    send_displs[i] = displs[i] * static_cast<int>(row_width);
  }

  MPI_Scatterv(full_matrix.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, local_matrix.data(),
               static_cast<int>(local_matrix.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // -------- Прямой ход --------
  std::vector<double> pivot_row(row_width);

  for (std::size_t k = 0; k < size; ++k) {
    // const int owner = static_cast<int>(k * proc_count / size);
    const int owner = FindOwner(k, displs, rows_per_proc);

    if (rank == owner) {
      const std::size_t local_k = k - static_cast<std::size_t>(displs[rank]);

      double pivot = local_matrix[(local_k * row_width) + k];

      for (std::size_t col = k; col < row_width; ++col) {
        pivot_row[col] = local_matrix[(local_k * row_width) + col] / pivot;
      }
    }

    MPI_Bcast(pivot_row.data(), static_cast<int>(row_width), MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for (std::size_t row = 0; row < local_rows; ++row) {
      const std::size_t global_row = static_cast<std::size_t>(displs[rank]) + row;

      if (global_row > k) {
        const double factor = local_matrix[(row * row_width) + k];

        for (std::size_t col = k; col < row_width; ++col) {
          local_matrix[(row * row_width) + col] -= factor * pivot_row[col];
        }
      }
    }
  }

  // -------- Сбор матрицы --------
  if (rank == 0) {
    full_matrix.resize(size * row_width);
  }

  MPI_Gatherv(local_matrix.data(), static_cast<int>(local_matrix.size()), MPI_DOUBLE, full_matrix.data(),
              send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // -------- Обратный ход (rank 0) --------
  if (rank == 0) {
    std::vector<double> solution(size, 0.0);

    for (std::size_t i = size; i-- > 0;) {
      double value = full_matrix[(i * row_width) + size];
      for (std::size_t j = i + 1; j < size; ++j) {
        value -= full_matrix[(i * row_width) + j] * solution[j];
      }
      solution[i] = value;
    }

    const double sum = std::accumulate(solution.begin(), solution.end(), 0.0);

    GetOutput() = static_cast<int>(std::round(std::abs(sum)));
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
