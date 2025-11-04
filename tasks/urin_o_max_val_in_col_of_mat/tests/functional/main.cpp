#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace urin_o_max_val_in_col_of_mat {

class UrinOMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    
    input_data_.clear();
    expected_output_.clear();
    
    // Создаем тестовую матрицу и вычисляем ожидаемые максимумы по столбцам
    for (int i = 0; i < matrix_size; ++i) {
      std::vector<int> row;
      for (int j = 0; j < matrix_size; ++j) {
        // Заполняем матрицу так, чтобы максимумы по столбцам были предсказуемы
        // В столбце j максимальное значение будет: (matrix_size - 1) * matrix_size + j
        row.push_back(i * matrix_size + j);
      }
      input_data_.push_back(row);
    }
    
    // Вычисляем ожидаемые максимумы по столбцам
    for (int j = 0; j < matrix_size; ++j) {
      int max_val = input_data_[0][j];
      for (int i = 1; i < matrix_size; ++i) {
        if (input_data_[i][j] > max_val) {
          max_val = input_data_[i][j];
        }
      }
      expected_output_.push_back(max_val);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(UrinOMaxValInColOfMatFuncTests, MaxValInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(2, "2x2_matrix"),
    std::make_tuple(3, "3x3_matrix"), 
    std::make_tuple(4, "4x4_matrix")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<UrinOMaxValInColOfMatMPI, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat),
                   ppc::util::AddFuncTask<UrinOMaxValInColOfMatSEQ, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = UrinOMaxValInColOfMatFuncTests::PrintFuncTestName<UrinOMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MaxValInColTests, UrinOMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace urin_o_max_val_in_col_of_mat
