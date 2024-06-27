#ifndef CONTACT_BENCH_BINDINGS_CONTEXT_H
#define CONTACT_BENCH_BINDINGS_CONTEXT_H

#include "contactbench/fwd.hpp"

namespace contactbench {
namespace context {

using T = double;

CONTACTBENCH_EIGEN_TYPEDEFS(T);

using FrictionConstraint = contactbench::FrictionConstraint<T>;

using IceCreamCone = contactbench::IceCreamCone<T>;

using PyramidCone = contactbench::PyramidCone<T>;

using DelassusPinocchio = contactbench::DelassusPinocchio<T>;

using DelassusDense = contactbench::DelassusDense<T>;

using Statistics = contactbench::Statistics<T>;

using ContactSolverSettings = contactbench::ContactSolverSettings<T>;

template <template <typename> class C>
using ContactProblem = contactbench::ContactProblem<T, C>;

template <template <typename> class C>
using BaseSolver = contactbench::BaseSolver<T, C>;

using CCPBaseSolver = contactbench::CCPBaseSolver<T>;

template <template <typename> class C>
using NCPPGSSolver = contactbench::NCPPGSSolver<T, C>;

using LCPQPSolver = contactbench::LCPQPSolver<T>;

template <template <typename> class C>
using NCPStagProjSolver = contactbench::NCPStagProjSolver<C>;

using CCPPGSSolver = contactbench::CCPPGSSolver<T>;

using CCPADMMSolver = contactbench::CCPADMMSolver<T>;

using CCPADMMPrimalSolver = contactbench::CCPADMMPrimalSolver<T>;

using CCPNewtonPrimalSolver = contactbench::CCPNewtonPrimalSolver<T>;

using RaisimSolver = contactbench::RaisimSolver<T>;
using RaisimCorrectedSolver = contactbench::RaisimCorrectedSolver<T>;

} // namespace context
} // namespace contactbench
#endif
