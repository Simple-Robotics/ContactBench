#pragma once

#if __cplusplus >= 201703L
#define CONTACTBENCH_WITH_CPP_17
#endif

#if __cplusplus >= 201402L
#define CONTACTBENCH_WITH_CPP_14
#endif

#ifdef CONTACTBENCH_EIGEN_CHECK_MALLOC
#define CONTACTBENCH_EIGEN_ALLOW_MALLOC(allowed)                               \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#else
#define CONTACTBENCH_EIGEN_ALLOW_MALLOC(allowed)
#endif

/// @brief Entering performance-critical code.
#define CONTACTBENCH_NOMALLOC_BEGIN CONTACTBENCH_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define CONTACTBENCH_NOMALLOC_END CONTACTBENCH_EIGEN_ALLOW_MALLOC(true)

#define CONTACTBENCH_INLINE inline __attribute__((always_inline))

/// @brief Unused function parameters
#define CONTACTBENCH_UNUSED(x) (void)(x)


/// @brief Macro to static-cast to size_t i.e unsigned long
#define CAST_UL(i) static_cast<size_t>(i)

#ifdef CONTACTBENCH_WITH_CPP_17
#define CONTACTBENCH_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define CONTACTBENCH_MAYBE_UNUSED
#else
#define CONTACTBENCH_MAYBE_UNUSED __attribute__((__unused__))
#endif

#define CONTACTBENCH_RUNTIME_ERROR(msg) throw std::runtime_error(msg)

#define CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(T)                        \
  std::vector<T, Eigen::aligned_allocator<T>>
