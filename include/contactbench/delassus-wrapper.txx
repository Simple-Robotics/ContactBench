#pragma once

#include "contactbench/delassus-wrapper.hpp"

namespace contactbench {

extern template struct DelassusBase<context::Scalar>;
extern template struct DelassusDense<context::Scalar>;
extern template struct DelassusPinocchio<context::Scalar>;

} // namespace contactbench
