#include <gtest/gtest.h>

/*#include <algorithm>
#include <cstddef>*/
#include <string>
#include <vector>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace urin_o_max_val_in_col_of_mat {

class UrinOMaxValInColOfMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int matrix_size = 50;
    test_matrix_.resize(matrix_size, std::vector<int>(matrix_size));

    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        test_matrix_[i][j] = ((i + j) % 100) + 1;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == test_matrix_[0].size();
  }

  InType GetTestInputData() final {
    return test_matrix_;
  }

 public:
  static std::string CustomPerfTestName(const testing::TestParamInfo<BaseRunPerfTests::ParamType> &info) {
    return "PerfTest_" + std::to_string(info.index);
  }

 private:
  std::vector<std::vector<int>> test_matrix_;
};

TEST_P(UrinOMaxValInColOfMatPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, UrinOMaxValInColOfMatMPI, UrinOMaxValInColOfMatSeq>(
    PPC_SETTINGS_urin_o_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = UrinOMaxValInColOfMatPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, UrinOMaxValInColOfMatPerfTests, kGtestValues, kPerfTestName);

}  // namespace urin_o_max_val_in_col_of_mat
