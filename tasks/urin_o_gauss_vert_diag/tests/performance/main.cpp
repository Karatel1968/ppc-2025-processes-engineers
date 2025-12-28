#include <gtest/gtest.h>
/*#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>*/
#include <cmath>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
#include "urin_o_gauss_vert_diag/mpi/include/ops_mpi.hpp"
#include "urin_o_gauss_vert_diag/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace urin_o_gauss_vert_diag {

class UrinRunPerfTestGaussVertical : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  UrinRunPerfTestGaussVertical() = default;

 protected:
  void SetUp() override {
    input_data_ = kMatrixSize;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    // std::cout << "CheckTestOutputData: output_data = " << output_data << std::endl;

    if (output_data <= 0) {
      // std::cout << "CheckTestOutputData: FAILED - output_data <= 0" << std::endl;
      return false;
    }

    // std::cout << "CheckTestOutputData: PASSED" << std::endl;
    // return true;
    return std::abs(output_data) > 1e-6;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  static constexpr InType kMatrixSize = 1000;
  InType input_data_{0};
};

TEST_P(UrinRunPerfTestGaussVertical, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, UrinOGaussVertDiagMPI, UrinOGaussVertDiagSEQ>(
    PPC_SETTINGS_urin_o_gauss_vert_diag);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, UrinRunPerfTestGaussVertical, kGtestValues,
                         UrinRunPerfTestGaussVertical::CustomPerfTestName);

}  // namespace urin_o_gauss_vert_diag
