#include "urin_o_gauss_vert_diag/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>  // для std::cerr
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

  // std::mt19937 gen(42);
  std::seed_seq seed{42, 12345};
  std::mt19937 gen(seed);
  // static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> off_diag(0.01, 0.1);
  std::uniform_real_distribution<double> diag_add(1.0, 2.0);  // Было (1.0, 5.0)
  std::uniform_real_distribution<double> rhs_dist(0.1, 1.0);

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
    // const std::size_t begin = static_cast<std::size_t>(displs[i]);
    const auto begin = static_cast<std::size_t>(displs[i]);
    const std::size_t end = begin + static_cast<std::size_t>(rows_per_proc[i]);
    if (global_row >= begin && global_row < end) {
      return static_cast<int>(i);
    }
  }
  return 0;
}

void UrinOGaussVertDiagMPI::EliminateLocalRows(std::vector<double> &local, const std::vector<double> &pivot_row,
                                               std::size_t local_rows, std::size_t width, std::size_t k, int rank,
                                               const std::vector<int> &displs) {
  for (std::size_t row = 0; row < local_rows; ++row) {
    const auto global_row = static_cast<std::size_t>(displs[rank]) + row;
    if (global_row > k) {
      const double factor = local[(row * width) + k];
      for (std::size_t col = k; col < width; ++col) {
        local[(row * width) + col] -= factor * pivot_row[col];
      }
    }
  }
}

void UrinOGaussVertDiagMPI::NormalizePivotRow(std::vector<double> &local, std::vector<double> &pivot_row,
                                              std::size_t local_k, std::size_t k, std::size_t width) {
  const double pivot = local[(local_k * width) + k];
  /*for (std::size_t col = k; col < width; ++col) {
    pivot_row[col] = local[(local_k * width) + col] / pivot;
  }*/
  for (std::size_t col = k; col < width; ++col) {
    local[(local_k * width) + col] /= pivot;
    pivot_row[col] = local[(local_k * width) + col];
  }
}

void UrinOGaussVertDiagMPI::DistributeRows(int proc_count, std::size_t size, std::vector<int> &rows_per_proc,
                                           std::vector<int> &displs) {
  for (int i = 0; i < proc_count; ++i) {
    rows_per_proc[i] = static_cast<int>(size / proc_count);
    if (static_cast<std::size_t>(i) < size % proc_count) {
      ++rows_per_proc[i];
    }
  }
  // std::partial_sum(rows_per_proc.begin(), rows_per_proc.end() - 1, displs.begin() + 1);
  for (int i = 1; i < proc_count; ++i) {
    displs[i] = displs[i - 1] + rows_per_proc[i - 1];
  }
}

OutType UrinOGaussVertDiagMPI::BackSubstitutionMPI(const std::vector<double> &full_matrix, std::size_t size,
                                                   std::size_t width) {
  std::vector<double> x(size, 0.0);

  for (int i = static_cast<int>(size) - 1; i >= 0; --i) {
    double s = full_matrix[(i * width) + size];

    for (std::size_t j = i + 1; j < size; ++j) {
      s -= full_matrix[(i * width) + j] * x[j];
    }

    // диагональ = 1, но оставим защиту
    x[i] = s / full_matrix[(i * width) + i];
  }

  double norm = 0.0;
  for (double v : x) {
    norm += std::abs(v);
  }

  return static_cast<OutType>(std::round(norm));
}

bool UrinOGaussVertDiagMPI::RunImpl() {
  int rank = 0;
  int proc_count = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  const auto size = static_cast<std::size_t>(GetInput());
  const std::size_t row_width = size + 1;

  std::vector<int> rows_per_proc(proc_count, 0);
  std::vector<int> displs(proc_count, 0);
  DistributeRows(proc_count, size, rows_per_proc, displs);
  // const std::size_t local_rows = static_cast<std::size_t>(rows_per_proc[rank]);

  const auto local_rows = static_cast<std::size_t>(rows_per_proc[rank]);

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

  if (local_matrix.size() != static_cast<std::size_t>(send_counts[rank])) {
    std::cerr << "Rank " << rank << ": local_matrix size = " << local_matrix.size()
              << ", but send_counts = " << send_counts[rank] << "\n";
    return false;
  }

  MPI_Scatterv(full_matrix.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, local_matrix.data(),
               send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // -------- Прямой ход --------
  std::vector<double> pivot_row(row_width, 0.0);

  for (std::size_t k = 0; k < size; ++k) {
    // const int owner = static_cast<int>(k * proc_count / size);
    // std::fill(pivot_row.begin(), pivot_row.end(), 0.0);
    std::ranges::fill(pivot_row, 0.0);
    const int owner = FindOwner(k, displs, rows_per_proc);

    if (rank == owner) {
      const auto local_k = k - static_cast<std::size_t>(displs[rank]);
      NormalizePivotRow(local_matrix, pivot_row, local_k, k, row_width);
    }

    MPI_Bcast(pivot_row.data(), static_cast<int>(row_width), MPI_DOUBLE, owner, MPI_COMM_WORLD);

    EliminateLocalRows(local_matrix, pivot_row, local_rows, row_width, k, rank, displs);
  }

  // -------- Сбор матрицы --------
  if (rank == 0) {
    full_matrix.resize(size * row_width);
  }

  MPI_Gatherv(local_matrix.data(), send_counts[rank], MPI_DOUBLE, rank == 0 ? full_matrix.data() : nullptr,
              send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // -------- Обратный ход (rank 0) --------
  OutType final_output = 0;  // BackSubstitutionMPI(rank, full_matrix, size, row_width);

  if (rank == 0) {
    final_output = BackSubstitutionMPI(full_matrix, size, row_width);
  }

  MPI_Bcast(&final_output, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::cout << "Rank " << rank << ": GetOutput() = " << final_output << "\n";

  GetOutput() = final_output;

  return true;
}

bool UrinOGaussVertDiagMPI::PostProcessingImpl() {
  return true;
}

}  // namespace urin_o_gauss_vert_diag
