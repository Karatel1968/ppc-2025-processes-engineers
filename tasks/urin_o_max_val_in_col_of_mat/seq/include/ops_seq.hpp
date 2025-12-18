#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "urin_o_max_val_in_col_of_mat/common/include/common.hpp"

namespace urin_o_max_val_in_col_of_mat {

class UrinOMaxValInColOfMatSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  using InType = std::vector<std::vector<int>>;
  using OutType = std::vector<int>;

  explicit UrinOMaxValInColOfMatSeq(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace urin_o_max_val_in_col_of_mat
