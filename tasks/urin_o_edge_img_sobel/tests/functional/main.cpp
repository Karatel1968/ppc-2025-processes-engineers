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

#include "urin_o_edge_img_sobel/common/include/common.hpp"
#include "urin_o_edge_img_sobel/mpi/include/ops_mpi.hpp"
#include "urin_o_edge_img_sobel/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace urin_o_edge_img_sobel {

class UrinOEdgeImgSobelFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  /*static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }*/
  static std::string PrintTestParam(const TestType &test_param) {
    std::string description = std::get<1>(test_param);
    return description;
  }
  /*static std::string PrintTestParam(const testing::TestParamInfo<
    std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>, std::string, TestType>>& info)
{ const auto& param = std::get<2>(info.param);  // TestType return std::to_string(std::get<0>(param)) + "_" +
std::get<1>(param);
}*/

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_urin_o_edge_img_sobel, "pic.jpg");
    auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 1);  // 1 = force grayscale
    if (!data) {
      throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }

    if (width != height) {
      stbi_image_free(data);
      throw std::runtime_error("Image width != height");
    }

    std::vector<int> pixels(width * height);
    for (int i = 0; i < width * height; i++) {
      pixels[i] = static_cast<int>(data[i]);
    }
    stbi_image_free(data);

    input_data_ = std::make_tuple(pixels, height, width);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что размер совпадает
    const std::size_t expected_size =
        static_cast<std::size_t>(std::get<1>(input_data_)) * static_cast<std::size_t>(std::get<2>(input_data_));

    if (output_data.size() != expected_size) {
      return false;
    }
    // Можно добавить более сложную проверку (напр. максимальные значения, средний порог и т.п.)
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(UrinOEdgeImgSobelFuncTests, SobelFromPic) {
  ExecuteTest(GetParam());
}

// Можно добавить несколько тестов с разными параметрами (например, порогами)
const std::array<TestType, 7> kTestParam = {
    std::make_tuple(3, "Test_3"),   std::make_tuple(5, "Test_5"), std::make_tuple(7, "Test_7"),
    std::make_tuple(8, "Test_8"),   std::make_tuple(9, "Test_9"), std::make_tuple(10, "Test_10"),
    std::make_tuple(11, "Test_11"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<UrinOEdgeImgSobelMPI, InType>(kTestParam, PPC_SETTINGS_urin_o_edge_img_sobel),
    ppc::util::AddFuncTask<UrinOEdgeImgSobelSEQ, InType>(kTestParam, PPC_SETTINGS_urin_o_edge_img_sobel));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

// const auto kPerfTestName = UrinOEdgeImgSobelFuncTests::PrintTestParam;
const auto kPerfTestName = UrinOEdgeImgSobelFuncTests::PrintFuncTestName<UrinOEdgeImgSobelFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, UrinOEdgeImgSobelFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace urin_o_edge_img_sobel
