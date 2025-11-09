#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatMPI::UrinOMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() =  OutType{};
}

bool UrinOMaxValInColOfMatMPI::ValidationImpl() {
  const auto& matrix = GetInput();
  
  if (matrix.empty()) {
    return false;
  }
  
  int rows = matrix.size();
  int cols = matrix[0].size();
  
  for (int i = 1; i < rows; ++i) {
    if (matrix[i].size() != static_cast<size_t>(cols)) {
      return false;
    }
  }
  
  return true;
}

bool UrinOMaxValInColOfMatMPI::PreProcessingImpl() {
  
  return true;
}

bool UrinOMaxValInColOfMatMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  const auto& matrix = GetInput();
  int rows = matrix.size();
  int cols = matrix[0].size();
  
  // Используем утилиту для получения количества потоков
  //const int num_threads = ppc::util::GetNumThreads();
  
  // Распределяем столбцы матрицы между процессами
  int base_cols_per_process = cols / size;
  int remainder = cols % size;
  
  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    start_col += base_cols_per_process + (i < remainder ? 1 : 0);
  }
  int end_col = start_col + base_cols_per_process + (rank < remainder ? 1 : 0);
  int local_cols_count = end_col - start_col;
  
  // Каждый процесс находит максимумы в своих столбцах
  std::vector<int> local_maxes(local_cols_count, 0);
  
  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    int global_col = start_col + local_idx;
    int max_val = matrix[0][global_col];
    
    for (int row = 1; row < rows; ++row) {
      if (matrix[row][global_col] > max_val) {
        max_val = matrix[row][global_col];
      }
    }
    local_maxes[local_idx] = max_val;
  }
  
  // Собираем результаты
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  
  MPI_Allgather(&local_cols_count, 1, MPI_INT, 
                recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i-1] + recv_counts[i-1];
  }
  
  // На процессе 0 создаем вектор для всех максимумов
  OutType all_column_maxes;
  if (rank == 0) {
    all_column_maxes.resize(cols);
  }
  
  MPI_Gatherv(local_maxes.data(), local_cols_count, MPI_INT,
              all_column_maxes.data(), recv_counts.data(), displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);
  
  // На процессе 0 устанавливаем результат
  /*if (rank == 0) {
    GetOutput() = all_column_maxes;
  }
  // Синхронизируем все процессы
  MPI_Barrier(MPI_COMM_WORLD);*/
  int result_size = 0;
if (rank == 0) {
    result_size = static_cast<int>(all_column_maxes.size());
}
MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (rank != 0) {
    all_column_maxes.resize(result_size);
}
MPI_Bcast(all_column_maxes.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);

// Установить результат на всех процессах
GetOutput() = all_column_maxes;
  // Все процессы должны вернуть true для успешного выполнения
  return true;
}



bool UrinOMaxValInColOfMatMPI::PostProcessingImpl() {
  const auto& output = GetOutput();
  
  // Простая проверка - результат не должен быть пустым
  return !output.empty();
}


}  // namespace urin_o_max_val_in_col_of_mat
//