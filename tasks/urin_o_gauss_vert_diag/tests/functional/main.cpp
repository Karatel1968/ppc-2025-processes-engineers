#include <gtest/gtest.h>
#include <mpi.h>
// #include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
#include "urin_o_gauss_vert_diag/mpi/include/ops_mpi.hpp"
#include "urin_o_gauss_vert_diag/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace urin_o_gauss_vert_diag {

class UrinRunFuncTestsGaussVertical : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // UrinRunFuncTestsGaussVertical() : input_data_(0), expected_output_(0) {}
  UrinRunFuncTestsGaussVertical() = default;

  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    test_name_ = std::get<1>(params);

    input_data_ = matrix_size;
    expected_output_ = 1;  // Ожидаем положительный результат для успешного решения
  }

  auto CheckTestOutputData(OutType &output_data) -> bool final {
    int mpi_initialized = 0;

    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // Синхронизируем результат между процессами
      OutType global_output = output_data;
      MPI_Bcast(&global_output, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Используем синхронизированное значение
      output_data = global_output;
    }
    return (output_data > 0);
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_{0};
  OutType expected_output_{0};
  std::string test_name_;
};

namespace {

TEST_P(UrinRunFuncTestsGaussVertical, GaussVerticalDiagonalTest) {
  // ExecuteTest(GetParam());
  try {
    ExecuteTest(GetParam());
  } catch (const std::exception &e) {
    std::cerr << "Exception in test: " << e.what() << "\n";
    throw;
  }
}

// Тестовые параметры: {размер_матрицы, название_теста}
const std::array<TestType, 6> kTestParam = {std::make_tuple(3, "small_matrix"),    std::make_tuple(5, "medium_matrix"),
                                            std::make_tuple(10, "large_matrix"),   std::make_tuple(15, "xlarge_matrix"),
                                            std::make_tuple(20, "xxlarge_matrix"), std::make_tuple(25, "huge_matrix")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<UrinOGaussVertDiagMPI, InType>(kTestParam, PPC_SETTINGS_urin_o_gauss_vert_diag),
    ppc::util::AddFuncTask<UrinOGaussVertDiagSEQ, InType>(kTestParam, PPC_SETTINGS_urin_o_gauss_vert_diag));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = UrinRunFuncTestsGaussVertical::PrintFuncTestName<UrinRunFuncTestsGaussVertical>;

INSTANTIATE_TEST_SUITE_P(GaussVerticalTests, UrinRunFuncTestsGaussVertical, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace urin_o_gauss_vert_diag
