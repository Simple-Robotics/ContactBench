#include "contactbench/solvers.hpp"

namespace contactbench {

template struct BaseSolver<context::Scalar, IceCreamCone>;
template struct BaseSolver<context::Scalar, PyramidCone>;
template struct DualBaseSolver<context::Scalar, IceCreamCone>;
template struct DualBaseSolver<context::Scalar, PyramidCone>;
template struct NCPPGSSolver<context::Scalar, IceCreamCone>;
template struct NCPPGSSolver<context::Scalar, PyramidCone>;
template struct LCPQPSolver<context::Scalar>;
template struct NCPStagProjSolver<IceCreamCone>;
template struct NCPStagProjSolver<PyramidCone>;
template struct CCPBaseSolver<context::Scalar>;
template struct CCPPGSSolver<context::Scalar>;
template struct CCPADMMSolver<context::Scalar>;
template struct CCPADMMPrimalSolver<context::Scalar>;
template struct CCPNewtonPrimalSolver<context::Scalar>;
template struct RaisimSolver<context::Scalar>;
template struct RaisimCorrectedSolver<context::Scalar>;

} // namespace contactbench
