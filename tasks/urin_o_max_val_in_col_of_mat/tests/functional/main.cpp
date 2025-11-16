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
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    return std::all_of(output_data.begin(), output_data.end(), [](int val) { return val > 0; });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(UrinOMaxValInColOfMatFuncTests, MaxValInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "10x10"), std::make_tuple(50, "50x50"),
                                            std::make_tuple(100, "100x100")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<UrinOMaxValInColOfMatMPI, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat),
    ppc::util::AddFuncTask<UrinOMaxValInColOfMatSeq, InType>(kTestParam, PPC_SETTINGS_urin_o_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = UrinOMaxValInColOfMatFuncTests::PrintFuncTestName<UrinOMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MaxValInColTests, UrinOMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);

}  // rnamespace

}  // namespace urin_o_max_val_in_col_of_mat
