#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
/*#include "util/include/util.hpp"*/

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatMPI::UrinOMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool UrinOMaxValInColOfMatMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = false;
  int rows = 0, cols = 0;

  if (rank == 0) {
    const auto &matrix = GetInput();
    is_valid = !matrix.empty() && !matrix[0].empty();
    if (is_valid) {
      rows = matrix.size();
      cols = matrix[0].size();
      // Проверяем что матрица прямоугольная
      for (const auto &row : matrix) {
        if (row.size() != cols) {
          is_valid = false;
          break;
        }
      }
    }
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

  // Получаем размеры матрицы
  int rows = 0, cols = 0;
  if (rank == 0) {
    const auto &matrix = GetInput();
    rows = matrix.size();
    cols = matrix[0].size();
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Создаем локальную матрицу на всех процессах
  std::vector<std::vector<int>> local_matrix(rows, std::vector<int>(cols));

  // Рассылаем матрицу всем процессам построчно
  if (rank == 0) {
    const auto &source_matrix = GetInput();
    for (int i = 0; i < rows; ++i) {
      std::copy(source_matrix[i].begin(), source_matrix[i].end(), local_matrix[i].begin());
      // Рассылаем строку всем процессам
      MPI_Bcast(local_matrix[i].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    for (int i = 0; i < rows; ++i) {
      MPI_Bcast(local_matrix[i].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  // Распределяем столбцы между процессами
  int base_cols_per_process = cols / size;
  int remainder = cols % size;

  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    start_col += base_cols_per_process + (i < remainder ? 1 : 0);
  }
  int end_col = start_col + base_cols_per_process + (rank < remainder ? 1 : 0);
  int local_cols_count = end_col - start_col;

  // Вычисляем локальные максимумы для назначенных столбцов
  std::vector<int> local_maxima(local_cols_count);

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    int global_col = start_col + local_idx;
    int col_max = local_matrix[0][global_col];

    for (int row = 1; row < rows; ++row) {
      if (local_matrix[row][global_col] > col_max) {
        col_max = local_matrix[row][global_col];
      }
    }
    local_maxima[local_idx] = col_max;
  }

  // Собираем результаты со всех процессов
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  MPI_Allgather(&local_cols_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

  // Собираем все максимумы на процессе 0
  OutType all_maxima(cols);
  MPI_Gatherv(local_maxima.data(), local_cols_count, MPI_INT, all_maxima.data(), recv_counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  // Рассылаем результаты всем процессам
  MPI_Bcast(all_maxima.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = all_maxima;
  return true;
}

bool UrinOMaxValInColOfMatMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
//
