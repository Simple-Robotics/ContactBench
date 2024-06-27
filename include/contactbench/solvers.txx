#pragma once

#include "contactbench/solvers.hpp"

namespace contactbench {

extern template struct BaseSolver<context::Scalar, IceCreamCone>;
extern template struct BaseSolver<context::Scalar, PyramidCone>;
extern template struct DualBaseSolver<context::Scalar, IceCreamCone>;
extern template struct DualBaseSolver<context::Scalar, PyramidCone>;
extern template struct NCPPGSSolver<context::Scalar, IceCreamCone>;
extern template struct NCPPGSSolver<context::Scalar, PyramidCone>;
extern template struct LCPQPSolver<context::Scalar>;
extern template struct NCPStagProjSolver< IceCreamCone>;
extern template struct NCPStagProjSolver< PyramidCone>;
extern template struct CCPBaseSolver<context::Scalar>;
extern template struct CCPPGSSolver<context::Scalar>;
extern template struct CCPADMMSolver<context::Scalar>;
extern template struct CCPADMMPrimalSolver<context::Scalar>;
extern template struct CCPNewtonPrimalSolver<context::Scalar>;
extern template struct RaisimSolver<context::Scalar>;
extern template struct RaisimCorrectedSolver<context::Scalar>;


} // namespace contactbench
