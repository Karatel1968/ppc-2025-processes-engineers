#include <gtest/gtest.h>

#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"
#include "urin_o_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "urin_o_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace urin_o_max_val_in_col_of_mat {

class UrinOMaxValInColOfMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    return std::all_of(output_data.begin(), output_data.end(), [](int val) { return val >= 1 && val <= 1000; });
  }

  InType GetTestInputData() final {
    return input_data_;
  }
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
