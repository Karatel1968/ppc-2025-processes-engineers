#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatMPI::UrinOMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool UrinOMaxValInColOfMatMPI::ValidationImpl() {
  return GetInput() > 0 && GetInput() <= 10000;
}

bool UrinOMaxValInColOfMatMPI::PreProcessingImpl() {
  return true;
}

bool UrinOMaxValInColOfMatMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = GetInput();

  std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[i][j] = (i * n + j) % 1000 + 1;
    }
  }

  int base_cols_per_process = n / size;
  int remainder = n % size;

  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    start_col += base_cols_per_process + (i < remainder ? 1 : 0);
  }
  int end_col = start_col + base_cols_per_process + (rank < remainder ? 1 : 0);
  int local_cols_count = end_col - start_col;

  std::vector<int> local_maxes(local_cols_count, 0);

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    int global_col = start_col + local_idx;
    int max_val = matrix[0][global_col];

    for (int row = 1; row < n; ++row) {
      if (matrix[row][global_col] > max_val) {
        max_val = matrix[row][global_col];
      }
    }
    local_maxes[local_idx] = max_val;
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  MPI_Allgather(&local_cols_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

  OutType all_column_maxes(n);

  MPI_Gatherv(local_maxes.data(), local_cols_count, MPI_INT, all_column_maxes.data(), recv_counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(all_column_maxes.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = all_column_maxes;
  return true;
}

bool UrinOMaxValInColOfMatMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
//
