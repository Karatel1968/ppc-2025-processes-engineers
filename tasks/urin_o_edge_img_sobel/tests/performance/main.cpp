#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "urin_o_edge_img_sobel/common/include/common.hpp"
#include "urin_o_edge_img_sobel/mpi/include/ops_mpi.hpp"
#include "urin_o_edge_img_sobel/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace urin_o_edge_img_sobel {

class UrinOEdgeImgSobelPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const int size = 256;
    // std::vector<int> pixels(size * size, 128);  // Константное изображение (средний серый)
    const std::vector<int>::size_type total_pixels =
        static_cast<std::vector<int>::size_type>(size) * static_cast<std::vector<int>::size_type>(size);

    std::vector<int> pixels(total_pixels, 128);
    input_data_ = std::make_tuple(pixels, size, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что результат не пустой
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(UrinOEdgeImgSobelPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, UrinOEdgeImgSobelMPI, UrinOEdgeImgSobelSEQ>(PPC_SETTINGS_urin_o_edge_img_sobel);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, UrinOEdgeImgSobelPerfTest, kGtestValues,
                         UrinOEdgeImgSobelPerfTest::CustomPerfTestName);

}  // namespace urin_o_edge_img_sobel
