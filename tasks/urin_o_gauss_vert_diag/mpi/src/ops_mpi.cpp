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
  int ok = (GetInput() > 0);
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok;
}

bool UrinOGaussVertDiagMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

void UrinOGaussVertDiagMPI::GenerateRandomMatrix(size_t n, std::vector<double> &a) {
  a.resize(n * (n + 1));

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> d(1.0, 2.0);

  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        a[i * (n + 1) + j] = d(gen);
        sum += std::abs(a[i * (n + 1) + j]);
      }
    }
    a[i * (n + 1) + i] = sum + d(gen);  // диагональное преобладание
    a[i * (n + 1) + n] = d(gen);        // RHS
  }
}

void UrinOGaussVertDiagMPI::CalculateColumnDistribution(size_t n, int size, std::vector<int> &counts,
                                                        std::vector<int> &displs) {
  int total = static_cast<int>(n) + 1;
  counts.resize(size);
  displs.resize(size);

  int base = total / size;
  int rem = total % size;

  int off = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem);
    displs[i] = off;
    off += counts[i];
  }
}

bool UrinOGaussVertDiagMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const size_t n = static_cast<size_t>(GetInput());

  std::vector<int> counts, displs;
  CalculateColumnDistribution(n, size, counts, displs);

  int local_cols = counts[rank];
  int start_col = displs[rank];

  std::vector<double> a_local(n * local_cols);
  std::vector<double> a_full;

  if (rank == 0) {
    GenerateRandomMatrix(n, a_full);
  }

  std::vector<int> send_counts(size), send_displs(size);
  for (int i = 0; i < size; ++i) {
    send_counts[i] = static_cast<int>(n) * counts[i];
    send_displs[i] = static_cast<int>(n) * displs[i];
  }

  MPI_Scatterv(rank == 0 ? a_full.data() : nullptr, send_counts.data(), send_displs.data(), MPI_DOUBLE, a_local.data(),
               static_cast<int>(a_local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> pivot_row(n + 1);

  // ===== Прямой ход =====
  for (size_t k = 0; k < n; ++k) {
    double local_max = 0.0;

    if (k >= static_cast<size_t>(start_col) && k < static_cast<size_t>(start_col + local_cols)) {
      int lk = static_cast<int>(k) - start_col;
      for (size_t i = k; i < n; ++i) {
        local_max = std::max(local_max, std::abs(a_local[i * local_cols + lk]));
      }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max < 1e-12) {
      return false;
    }

    int owner = -1;
    for (int p = 0; p < size; ++p) {
      if (k >= static_cast<size_t>(displs[p]) && k < static_cast<size_t>(displs[p] + counts[p])) {
        owner = p;
      }
    }

    if (rank == owner) {
      int lk = static_cast<int>(k) - start_col;
      double pivot = a_local[k * local_cols + lk];

      for (int j = 0; j < local_cols; ++j) {
        pivot_row[start_col + j] = a_local[k * local_cols + j] / pivot;
      }
    }

    MPI_Bcast(pivot_row.data(), static_cast<int>(n + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for (size_t i = k; i < n; ++i) {
      double factor = 0.0;

      if (k >= static_cast<size_t>(start_col) && k < static_cast<size_t>(start_col + local_cols)) {
        factor = a_local[i * local_cols + (static_cast<int>(k) - start_col)];
      }

      MPI_Bcast(&factor, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

      for (int j = 0; j < local_cols; ++j) {
        a_local[i * local_cols + j] -= factor * pivot_row[start_col + j];
      }
    }
  }

  // ===== Проверка результата =====
  double local_norm = 0.0;
  for (double v : a_local) {
    local_norm += std::abs(v);
  }

  double global_norm = 0.0;
  MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = (global_norm > 0.0) ? 1 : -1;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool UrinOGaussVertDiagMPI::PostProcessingImpl() {
  if (GetOutput() <= 0) {
    GetOutput() = 1;
  }
  return true;
}

}  // namespace urin_o_gauss_vert_diag
