#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>
// #include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
/*#include "util/include/util.hpp"*/

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatSeq::UrinOMaxValInColOfMatSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // GetInput() = in;
  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType();  // Explicit empty vector
  }
  GetOutput() = OutType{};
}

bool UrinOMaxValInColOfMatSeq::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    return false;
  }

  size_t cols = matrix[0].size();
  /*for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;  // Не прямоугольная матрица
    }
  }

  return true;*/
  // return std::all_of(matrix.begin(), matrix.end(), [cols](const auto &row) { return row.size() == cols; });
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool UrinOMaxValInColOfMatSeq::PreProcessingImpl() {
  return true;
}

bool UrinOMaxValInColOfMatSeq::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = OutType();
    return false;
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  // Инициализируем выходной вектор размером с количество столбцов
  OutType column_maxima(cols);

  // Для каждого столбца находим максимальное значение
  for (size_t col = 0; col < cols; ++col) {
    int max_val = matrix[0][col];  // Начинаем с первого элемента столбца

    for (size_t row = 1; row < rows; ++row) {
      /*if (matrix[row][col] > max_val) {
        max_val = matrix[row][col];
      }*/
      max_val = std::max(matrix[row][col], max_val);
    }

    column_maxima[col] = max_val;
  }

  GetOutput() = column_maxima;
  return true;
}

bool UrinOMaxValInColOfMatSeq::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
