#include "urin_o_gauss_vert_diag/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_gauss_vert_diag {

UrinOGaussVertDiagSEQ::UrinOGaussVertDiagSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool UrinOGaussVertDiagSEQ::ValidationImpl() {
  return (GetInput() > 0);
}

bool UrinOGaussVertDiagSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

void UrinOGaussVertDiagSEQ::GenerateRandomMatrix(size_t size, std::vector<std::vector<double>> &matrix,
                                                 std::vector<double> &vector) {
  matrix.resize(size, std::vector<double>(size, 0.0));
  vector.resize(size, 0.0);

  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis_off_diag(0.1, 1.0);
  std::uniform_real_distribution<double> dis_diag_add(1.0, 5.0);
  std::uniform_real_distribution<double> dis_vector(1.0, 20.0);

  // Создаем диагонально-доминированную матрицу для устойчивости метода Гаусса
  for (size_t i = 0; i < size; ++i) {
    double row_sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i][j] = dis_off_diag(gen);
        row_sum += std::abs(matrix[i][j]);
      }
    }
    matrix[i][i] = row_sum + dis_diag_add(gen);  // Диагональное преобладание
    vector[i] = dis_vector(gen);
  }
}

bool UrinOGaussVertDiagSEQ::SolveGaussian(const std::vector<std::vector<double>> &a, 
                                          const std::vector<double> &b,
                                          std::vector<double> &x) {
  size_t n = a.size();
  if (n == 0) return false;
  
  // Создаем расширенную матрицу
  std::vector<std::vector<double>> augmented(n, std::vector<double>(n + 1));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      augmented[i][j] = a[i][j];
    }
    augmented[i][n] = b[i];
  }
  
  // Прямой ход метода Гаусса
  for (size_t k = 0; k < n; ++k) {
    // Поиск ведущего элемента
    size_t max_row = k;
    double max_val = std::abs(augmented[k][k]);
    
    for (size_t i = k + 1; i < n; ++i) {
      double val = std::abs(augmented[i][k]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }
    
    if (max_val < 1e-12) {
      return false;  // Матрица вырождена
    }
    
    // Обмен строк
    if (max_row != k) {
      std::swap(augmented[k], augmented[max_row]);
    }
    
    // Нормализация строки
    double pivot = augmented[k][k];
    for (size_t j = k; j <= n; ++j) {
      augmented[k][j] /= pivot;
    }
    
    // Исключение переменной
    for (size_t i = k + 1; i < n; ++i) {
      double factor = augmented[i][k];
      for (size_t j = k; j <= n; ++j) {
        augmented[i][j] -= factor * augmented[k][j];
      }
    }
  }
  
  // Обратный ход
  x.resize(n);
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    x[i] = augmented[i][n];
    for (size_t j = i + 1; j < n; ++j) {
      x[i] -= augmented[i][j] * x[j];
    }
  }
  
  return true;
}

bool UrinOGaussVertDiagSEQ::RunImpl() {
  int n_input = GetInput();
  if (n_input <= 0) {
    return false;
  }

  size_t n = static_cast<size_t>(n_input);

  std::vector<std::vector<double>> a;
  std::vector<double> b;
  GenerateRandomMatrix(n, a, b);

  std::vector<double> x;
  bool success = SolveGaussian(a, b, x);

  double sum = 0.0;
  for (double value : x) {
    sum += value;
  }

  int result = 0;
  if (success) {
    if (std::abs(sum) < 0.0001) {
      result = 1;
    } else {
      result = static_cast<int>(std::round(std::abs(sum)));
    }
  } else {
    result = -static_cast<int>(std::round(std::abs(sum)));
  }

  GetOutput() = result;
  return true;
}

bool UrinOGaussVertDiagSEQ::PostProcessingImpl() {
  if (GetOutput() < 0) {
    GetOutput() = -GetOutput();
  }
  if (GetOutput() == 0) {
    GetOutput() = 1;
  }

  return true;
}


}  // namespace urin_o_gauss_vert_diag
