#include <stdexcept>

#include "./macros.hpp"

// fwd-declaration of namespace
namespace contactbench {

// fwd FrictionConstraint
template <typename T> struct FrictionConstraint;

// fwd IceCreamCone
template <typename T> struct IceCreamCone;

// fwd PyramidCone
template <typename T> struct PyramidCone;

// fwd DelassusBase
template <typename T> struct DelassusBase;

// fwd DelassusPinocchio
template <typename T> struct DelassusPinocchio;

// fwd DelassusDense
template <typename T> struct DelassusDense;

// fwd ContactProblem
template <typename T, template <typename> class ConstraintTpl>
struct ContactProblem;

// fwd BaseSolver
template <typename T, template <typename> class ConstraintTpl>
struct BaseSolver;

// fwd NCPPGSSolver
template <typename T, template <typename> class ConstraintTpl>
struct NCPPGSSolver;

template <typename T> struct LCPQPSolver;

// fwd NCPStagProjSolver
template <template <typename> class ConstraintTpl> struct NCPStagProjSolver;

// fwd CCPBaseSolver
template <typename T> struct CCPBaseSolver;

// fwd CCPPGSSolver
template <typename T> struct CCPPGSSolver;
template <typename T> struct CCPADMMSolver;
template <typename T> struct CCPADMMPrimalSolver;
template <typename T> struct CCPNewtonPrimalSolver;

// fwd
template <typename T> struct RaisimSolver;
template <typename T> struct RaisimCorrectedSolver;

// fwd
template <typename T> struct Statistics;
template <typename T> struct ContactSolverSettings;

} // namespace contactbench

#include "contactbench/math.hpp"

#include "contactbench/context.hpp"
