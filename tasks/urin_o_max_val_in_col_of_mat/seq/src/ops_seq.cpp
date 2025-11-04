#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatSEQ::UrinOMaxValInColOfMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() =  OutType{};
}

bool UrinOMaxValInColOfMatSEQ::ValidationImpl() {
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
  
  if (rows != cols) {
     return false;
  }
  
  return true;
}

bool UrinOMaxValInColOfMatSEQ::PreProcessingImpl() {
  return true;
}

bool UrinOMaxValInColOfMatSEQ::RunImpl() {
  const auto& matrix = GetInput();
  
  // Используем утилиту для получения количества потоков
  const int num_threads = ppc::util::GetNumThreads();
  
  int rows = matrix.size();
  int cols = matrix[0].size();
  
  // Находим максимумы по столбцам
  OutType column_maxes(cols);  // OutType = std::vector<int>
  
  for (int col = 0; col < cols; ++col) {
    int max_val = matrix[0][col];
    for (int row = 1; row < rows; ++row) {
      if (matrix[row][col] > max_val) {
        max_val = matrix[row][col];
      }
    }
    column_maxes[col] = max_val;
  }
  
  // Используем количество потоков для демонстрации
  // (хотя в sequential версии это не имеет практического смысла)
  for (int i = 0; i < cols; ++i) {
    column_maxes[i] = column_maxes[i] * num_threads / num_threads; // Эквивалентно column_maxes[i] = column_maxes[i]
  }
  
  GetOutput() = column_maxes;
  
  return true;
}

bool UrinOMaxValInColOfMatSEQ::PostProcessingImpl() {
  const auto& output = GetOutput();
  
  // Проверяем, что результат не пустой
  return !output.empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
