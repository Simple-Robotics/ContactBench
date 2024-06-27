#include "contactbench/contact-problem.hpp"

namespace contactbench {

template struct DelassusBase<context::Scalar>;
template struct DelassusDense<context::Scalar>;
template struct DelassusPinocchio<context::Scalar>;

} // namespace contactbench
