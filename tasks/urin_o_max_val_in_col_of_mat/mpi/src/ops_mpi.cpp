#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

// #include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
/*#include "util/include/util.hpp"*/

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatMPI::UrinOMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // GetInput() = in;
  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType();  // Explicit empty vector
  }
  GetOutput() = std::vector<int>();
}

bool UrinOMaxValInColOfMatMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;
  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    const auto &matrix = this->GetInput();
    is_valid = !matrix.empty() && !matrix[0].empty();
    if (is_valid) {
      rows = static_cast<int>(matrix.size());
      cols = static_cast<int>(matrix[0].size());
      // Проверяем что матрица прямоугольная
      for (const auto &row : matrix) {
        // if (row.size() != static_cast<size_t>(cols)) {
        if (row.size() != static_cast<size_t>(cols)) {
          is_valid = false;
          break;
        }
      }
    }
    /*if (!is_valid) {
      rows = 0;
      cols = 0;
    }*/
  }

  // Рассылаем результат валидации и размеры матрицы
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid;
}

bool UrinOMaxValInColOfMatMPI::PreProcessingImpl() {
  return true;
}

bool UrinOMaxValInColOfMatMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto [rows, cols] = GetMatrixDimensions(rank);
  if (rows == 0 || cols == 0) {
    GetOutput() = OutType();
    return true;
  }

  std::vector<std::vector<int>> local_matrix(rows, std::vector<int>(cols));
  DistributeMatrixData(rank, rows, cols, local_matrix);
  auto column_dist = CalculateColumnDistribution(rank, size, cols);
  int start_col = column_dist.first;
  int local_cols_count = column_dist.second;

  // Compute local maxima
  auto local_maxima = ComputeLocalMaxima(local_matrix, rows, start_col, local_cols_count);

  /*auto [start_col, local_cols_count] = CalculateColumnDistribution(rank, size, cols);
  auto local_maxima = ComputeLocalMaxima(local_matrix, rows, start_col, local_cols_count);*/

  GetOutput() = GatherResults(local_maxima, size, cols);
  return true;
}

// Helper method 1: Get matrix dimensions
std::pair<int, int> UrinOMaxValInColOfMatMPI::GetMatrixDimensions(int rank) {
  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    const auto &matrix = this->GetInput();  // Явное использование this->
    if (matrix.empty() || matrix[0].empty()) {
      rows = 0;
      cols = 0;
    } else {
      rows = static_cast<int>(matrix.size());
      cols = static_cast<int>(matrix[0].size());
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return std::make_pair(rows, cols);
}

void UrinOMaxValInColOfMatMPI::DistributeMatrixData(int rank, int rows, int cols,
                                                    std::vector<std::vector<int>> &local_matrix) {
  if (rank == 0) {
    const auto &source_matrix = this->GetInput();  // Явное использование this->
    for (int i = 0; i < rows; ++i) {
      std::copy(source_matrix[i].begin(), source_matrix[i].end(), local_matrix[i].begin());
      MPI_Bcast(local_matrix[i].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    for (int i = 0; i < rows; ++i) {
      MPI_Bcast(local_matrix[i].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

std::pair<int, int> UrinOMaxValInColOfMatMPI::CalculateColumnDistribution(int rank, int size, int cols) {
  int base_cols_per_process = cols / size;
  int remainder = cols % size;

  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    start_col += base_cols_per_process + (i < remainder ? 1 : 0);
  }

  int local_cols_count = base_cols_per_process + (rank < remainder ? 1 : 0);
  return std::make_pair(start_col, local_cols_count);
}

std::vector<int> UrinOMaxValInColOfMatMPI::ComputeLocalMaxima(const std::vector<std::vector<int>> &matrix, int rows,
                                                              int start_col, int col_count) {
  std::vector<int> maxima(col_count);
  for (int i = 0; i < col_count; ++i) {
    int global_col = start_col + i;
    int max_val = matrix[0][global_col];
    for (int row = 1; row < rows; ++row) {
      max_val = std::max(matrix[row][global_col], max_val);
    }
    maxima[i] = max_val;
  }
  return maxima;
}

UrinOMaxValInColOfMatMPI::OutType UrinOMaxValInColOfMatMPI::GatherResults(const std::vector<int> &local_maxima,
                                                                          int size, int cols) {
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int local_count = static_cast<int>(local_maxima.size());

  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  OutType result(cols);
  MPI_Gatherv(local_maxima.data(), local_count, MPI_INT, result.data(), counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  MPI_Bcast(result.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
  return result;
}

bool UrinOMaxValInColOfMatMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
//
