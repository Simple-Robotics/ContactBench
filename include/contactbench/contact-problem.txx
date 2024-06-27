#pragma once

#include "contactbench/contact-problem.hpp"

namespace contactbench {

extern template struct ContactProblem<context::Scalar, IceCreamCone>;
extern template struct ContactProblem<context::Scalar, PyramidCone>;


} // namespace contactbench
