#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

UrinOMaxValInColOfMatSeq::UrinOMaxValInColOfMatSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool UrinOMaxValInColOfMatSeq::ValidationImpl() {
  return GetInput() > 0 && GetInput() <= 10000;
}

bool UrinOMaxValInColOfMatSeq::PreProcessingImpl() {
  return true;
}

bool UrinOMaxValInColOfMatSeq::RunImpl() {
  int n = GetInput();

  std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[i][j] = (i * n + j) % 1000 + 1;
    }
  }

  OutType column_maxes(n);

  for (int col = 0; col < n; ++col) {
    int max_val = matrix[0][col];
    for (int row = 1; row < n; ++row) {
      if (matrix[row][col] > max_val) {
        max_val = matrix[row][col];
      }
    }
    column_maxes[col] = max_val;
  }

  GetOutput() = column_maxes;
  return true;
}

bool UrinOMaxValInColOfMatSeq::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace urin_o_max_val_in_col_of_mat
