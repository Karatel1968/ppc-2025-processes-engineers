#include <gtest/gtest.h>
// #include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

static std::vector<std::vector<int>> GenerateTestMatrix(int size) {
  std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = ((i * size + j) % 1000) + 1;
    }
  }
  return matrix;
}

// Вспомогательная функция для вычисления ожидаемых результатов
static std::vector<int> CalculateExpectedMaxima(const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty()) {
    return {};
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();
  std::vector<int> expected_maxima(cols);

  for (size_t col = 0; col < cols; ++col) {
    int max_val = matrix[0][col];
    for (size_t row = 1; row < rows; ++row) {
      /*if (matrix[row][col] > max_val) {
        max_val = matrix[row][col];
      }*/
      max_val = std::max(matrix[row][col], max_val);
    }
    expected_maxima[col] = max_val;
  }

  return expected_maxima;
}

class UrinOMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_size_ = std::get<0>(params);
    test_matrix_ = GenerateTestMatrix(matrix_size_);
    expected_output_ = CalculateExpectedMaxima(test_matrix_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return test_matrix_;
  }

 private:
  int matrix_size_ = 0;
  std::vector<std::vector<int>> test_matrix_;
  std::vector<int> expected_output_;
};

namespace {

TEST_P(UrinOMaxValInColOfMatFuncTests, MaxValInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "tiny"),      std::make_tuple(2, "very_small"), std::make_tuple(3, "small"),
    std::make_tuple(4, "compact"),   std::make_tuple(5, "modest"),     std::make_tuple(6, "medium"),
    std::make_tuple(7, "moderate"),  std::make_tuple(8, "standard"),   std::make_tuple(9, "large"),
    std::make_tuple(10, "generous"),
};

const std::array<TestType, 4> kEdgeCaseTestParam = {
    std::make_tuple(1, "single_element"),
    std::make_tuple(2, "two_by_two"),
    std::make_tuple(100, "medium_large"),
    std::make_tuple(500, "large_matrix"),
};

class UrinOMaxValInColOfMatEdgeCaseTests : public UrinOMaxValInColOfMatFuncTests {};

TEST_P(UrinOMaxValInColOfMatEdgeCaseTests, EdgeCaseTests) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<UrinOMaxValInColOfMatMPI, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat),
    ppc::util::AddFuncTask<UrinOMaxValInColOfMatSeq, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat));

const auto kEdgeCaseTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<UrinOMaxValInColOfMatMPI, InType>(
                                                       kEdgeCaseTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat),
                                                   ppc::util::AddFuncTask<UrinOMaxValInColOfMatSeq, InType>(
                                                       kEdgeCaseTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kEdgeCaseGtestValues = ppc::util::ExpandToValues(kEdgeCaseTestTasksList);

const auto kPerfTestName = UrinOMaxValInColOfMatFuncTests::PrintFuncTestName<UrinOMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, UrinOMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(EdgeCaseTests, UrinOMaxValInColOfMatEdgeCaseTests, kEdgeCaseGtestValues, kPerfTestName);

}  // namespace
}  // namespace urin_o_max_val_in_col_of_mat
