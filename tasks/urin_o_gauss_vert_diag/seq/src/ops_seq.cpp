#include "urin_o_gauss_vert_diag/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
// #include <numeric>
#include <cstddef>
#include <random>
#include <vector>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace urin_o_gauss_vert_diag {

UrinOGaussVertDiagSEQ::UrinOGaussVertDiagSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool UrinOGaussVertDiagSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool UrinOGaussVertDiagSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

void UrinOGaussVertDiagSEQ::GenerateRandomMatrix(std::size_t size, std::vector<std::vector<double>> &matrix,
                                                 std::vector<double> &rhs) {
  /*matrix.assign(size, std::vector<double>(size, 0.0));
  rhs.assign(size, 0.0);*/
  matrix.clear();
  matrix.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    matrix[i].assign(size, 0.0);
  }

  rhs.clear();
  rhs.resize(size, 0.0);

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_real_distribution<double> off_diag_dist(0.1, 1.0);
  std::uniform_real_distribution<double> diag_dist(1.0, 5.0);
  std::uniform_real_distribution<double> rhs_dist(1.0, 20.0);

  for (std::size_t row = 0; row < size; ++row) {
    double row_sum = 0.0;

    for (std::size_t col = 0; col < size; ++col) {
      if (row != col) {
        const double value = off_diag_dist(generator);
        matrix[row][col] = value;
        row_sum += std::abs(value);
      }
    }

    matrix[row][row] = row_sum + diag_dist(generator);
    rhs[row] = rhs_dist(generator);
  }
}

void UrinOGaussVertDiagSEQ::EliminateRows(std::size_t pivot, std::vector<std::vector<double>> &augmented) {
  const std::size_t size = augmented.size();

  for (std::size_t row = pivot + 1; row < size; ++row) {
    const double factor = augmented[row][pivot];
    for (std::size_t col = pivot; col <= size; ++col) {
      augmented[row][col] -= factor * augmented[pivot][col];
    }
  }
}

void UrinOGaussVertDiagSEQ::BackSubstitution(const std::vector<std::vector<double>> &augmented,
                                             std::vector<double> &solution) {
  const std::size_t size = augmented.size();
  solution.assign(size, 0.0);

  for (std::ptrdiff_t row = static_cast<std::ptrdiff_t>(size) - 1; row >= 0; --row) {
    double value = augmented[row][size];
    for (std::size_t col = row + 1; col < size; ++col) {
      value -= augmented[row][col] * solution[col];
    }
    solution[row] = value;
  }
}

// namespace

bool UrinOGaussVertDiagSEQ::ForwardElimination(std::vector<std::vector<double>> &augmented) {
  const std::size_t size = augmented.size();

  for (std::size_t pivot = 0; pivot < size; ++pivot) {
    std::size_t max_row = pivot;
    double max_value = std::abs(augmented[pivot][pivot]);

    for (std::size_t row = pivot + 1; row < size; ++row) {
      const double value = std::abs(augmented[row][pivot]);
      if (value > max_value) {
        max_value = value;
        max_row = row;
      }
    }

    if (max_value < 1e-12) {
      return false;
    }

    if (max_row != pivot) {
      std::swap(augmented[pivot], augmented[max_row]);
    }

    const double divisor = augmented[pivot][pivot];
    for (std::size_t col = pivot; col <= size; ++col) {
      augmented[pivot][col] /= divisor;
    }

    EliminateRows(pivot, augmented);
  }

  return true;
}

bool UrinOGaussVertDiagSEQ::SolveGaussian(const std::vector<std::vector<double>> &matrix,
                                          const std::vector<double> &rhs, std::vector<double> &solution) {
  const auto size = matrix.size();
  if (size == 0 || rhs.size() != size) {
    return false;
  }

  std::vector<std::vector<double>> augmented(size, std::vector<double>(size + 1));

  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t col = 0; col < size; ++col) {
      augmented[row][col] = matrix[row][col];
    }
    augmented[row][size] = rhs[row];
  }

  if (!ForwardElimination(augmented)) {
    return false;
  }

  BackSubstitution(augmented, solution);
  return true;
}

bool UrinOGaussVertDiagSEQ::RunImpl() {
  const auto input_value = GetInput();
  const auto size = static_cast<std::size_t>(input_value);

  std::vector<std::vector<double>> matrix;
  std::vector<double> rhs;
  GenerateRandomMatrix(size, matrix, rhs);

  std::vector<double> solution;
  const bool success = SolveGaussian(matrix, rhs, solution);

  double sum = 0.0;
  for (const double value : solution) {
    sum += value;
  }

  if (!success) {
    GetOutput() = 1;
    return true;
  }

  GetOutput() = std::max(1, static_cast<int>(std::round(std::abs(sum))));
  return true;
}

bool UrinOGaussVertDiagSEQ::PostProcessingImpl() {
  if (GetOutput() <= 0) {
    GetOutput() = 1;
  }
  return true;
}

}  // namespace urin_o_gauss_vert_diag
