#pragma once

#include "urin_o_gauss_vert_diag/common/include/common.hpp"
#include "task/include/task.hpp"

namespace urin_o_gauss_vert_diag {

class UrinOGaussVertDiagMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit UrinOGaussVertDiagMPI(const InType &in);

  private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  static void GenerateRandomMatrix(size_t size, 
                                   std::vector<double> &vector);
  void CalculateColumnDistribution(size_t n, int size, std::vector<int> &col_counts,
                                   std::vector<int> &col_displs);
  void ScatterData(const std::vector<std::vector<double>> &a_full,
                   const std::vector<double> &b_full, std::vector<double> &a_local, const std::vector<int> &col_counts,
                   const std::vector<int> &col_displs, int size, size_t n);
  bool ReceiveScatteredData(std::vector<double> &a_local,
                            int local_cols, size_t n);
  bool CheckDiagonalElements(const std::vector<std::vector<double>> &a);
  bool CheckForErrors(bool has_error);
  bool SolveGaussian(std::vector<double> &a_local,
                     std::vector<double> &x_global, const std::vector<int> &col_counts,
                     const std::vector<int> &col_displs, int local_cols, 
                     int start_col, size_t n);
  double CalculateGlobalSum(const std::vector<double> &x_global);
  int CalculateResult(double global_sum, bool success, int rank);
  void BroadcastResult(int &result);
};

}  // namespace urin_o_gauss_vert_diag
