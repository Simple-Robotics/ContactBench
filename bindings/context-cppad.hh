#ifndef CONTACT_BENCH_BINDINGS_CONTEXT_CPPAD_H
#define CONTACT_BENCH_BINDINGS_CONTEXT_CPPAD_H

#include <pinocchio/autodiff/cppad.hpp>
#include "contactbench/fwd.hpp"

namespace contactbench {
namespace context {

using T = CppAD::AD<double>;

CONTACTBENCH_EIGEN_TYPEDEFS(T);

using IceCreamCone = contactbench::IceCreamCone<T>;
template <template <typename> class C>
using ContactProblem = contactbench::ContactProblem<T, C>;

template <template <typename> class C>
using BaseSolver = contactbench::BaseSolver<T, C>;

using CCPBaseSolver = contactbench::CCPBaseSolver<T>;

template <template <typename> class C>
using NPCPGSSolver = contactbench::NPCPGSSolver<C>;

} // namespace context
} // namespace contactbench
#endif
