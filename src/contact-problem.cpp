#include "contactbench/contact-problem.hpp"

namespace contactbench {

template struct ContactProblem<context::Scalar, IceCreamCone>;
template struct ContactProblem<context::Scalar, PyramidCone>;

} // namespace contactbench
