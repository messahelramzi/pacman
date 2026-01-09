#include <chrono>
#include <iostream>

#include "common/types.hpp"
#include "interpolator.hpp"
#include "utils.hpp"

#undef VIEW_ALLOC

#define VIEW_ALLOC(name)                                                       \
  Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace{},                      \
                     Kokkos::WithoutInitializing, _region_name + "::" + name)

template <typename ExecSpace, typename RbfFunc, int_t Dim>
IterationResult _run_test(const std::string &src, const std::string &dst,
                          const index_t iterations) {
  const std::string _region_name = "test";
  index_t N, M;
  auto source_grid = get_points_from_vtu_grid<Dim>(src.c_str(), &N);
  auto target_grid = get_points_from_vtu_grid<Dim>(dst.c_str(), &M);

  using PointsView = Kokkos::View<ArborX::Point<Dim, coordinates_t> *,
                                  Kokkos::DefaultHostExecutionSpace>;
  using ScalarsView = Kokkos::View<fp_t *, Kokkos::DefaultHostExecutionSpace>;

  Kokkos::DefaultHostExecutionSpace hostspace{};

  PointsView source(VIEW_ALLOC("source"), N);
  ScalarsView values(VIEW_ALLOC("values"), N);
  PointsView target(VIEW_ALLOC("target"), M);
  ScalarsView reference(VIEW_ALLOC("reference"), M);
  ScalarsView errors(VIEW_ALLOC("errors"), M);

  Kokkos::parallel_for("test::fill source and values",
                       Kokkos::RangePolicy(hostspace, 0, N),
                       [&](const index_t &i) {
                         source(i) = source_grid[i];
                         values(i) = franke_function(source_grid[i]);
                       });
  Kokkos::parallel_for("test::fill reference",
                       Kokkos::RangePolicy(hostspace, 0, M),
                       [&](const index_t &i) {
                         target(i) = target_grid[i];
                         reference(i) = franke_function(target_grid[i]);
                       });
  Kokkos::fence();

  RbfFunc rbf_function{};
  std::vector<double> ts;
  const auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch();
  auto interpolator =
      RbfPumInterpolator<ExecSpace, Dim, decltype(rbf_function)>(
          source, values, target, rbf_function, 50, 0.15, 0.1);
  const auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();
  for (index_t it = 0; it < iterations; ++it) {
    const auto begin =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    auto interpolator2 =
        RbfPumInterpolator<ExecSpace, Dim, decltype(rbf_function)>(
            source, values, target, rbf_function, 50, 0.15, 0.1);
    const auto end =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    ts.push_back((t2 - t1).count() / 1'000'000.0);
  }

  Kokkos::fence();
  auto interpolated_data =
      Kokkos::create_mirror_view_and_copy(hostspace, interpolator.out);

  Kokkos::parallel_for(
      "compute abs errors", Kokkos::RangePolicy(hostspace, 0, M),
      [&](const index_t &i) {
        errors(i) = Kokkos::fabs(reference(i) - interpolated_data(i));
      });
  Kokkos::fence();

  Kokkos::sort(errors);
  fp_t sum = 0.0;
  Kokkos::parallel_reduce(
      "sum", Kokkos::RangePolicy(hostspace, 0, M),
      [&](const index_t i, fp_t &update) { update += errors(i); }, sum);
  fp_t mean = sum / M;

  fp_t min_val, max_val;
  Kokkos::deep_copy(min_val, Kokkos::subview(errors, 0));
  Kokkos::deep_copy(max_val, Kokkos::subview(errors, errors.extent(0) - 1));

  auto worst_idx = [&](double pct) {
    index_t i = static_cast<index_t>(pct * M);
    if (i >= M)
      i = M - 1;
    return i;
  };

  index_t i1 = worst_idx(0.99);
  index_t i5 = worst_idx(0.95);
  index_t i10 = worst_idx(0.90);
  index_t i50 = worst_idx(0.50);

  fp_t worst1, worst5, worst10, worst50;
  Kokkos::deep_copy(worst1, Kokkos::subview(errors, i1));
  Kokkos::deep_copy(worst5, Kokkos::subview(errors, i5));
  Kokkos::deep_copy(worst10, Kokkos::subview(errors, i10));
  Kokkos::deep_copy(worst50, Kokkos::subview(errors, i50));

  IterationResult res{ITERATIONRESULT_UNDEFINED_ID,
                      "",
                      mean,
                      min_val,
                      max_val,
                      worst50,
                      worst10,
                      worst5,
                      worst1,
                      std::accumulate(ts.begin(), ts.end(), 0.0) /
                          static_cast<double>(iterations)};

  free(source_grid);
  free(target_grid);

  return res;
}

template <int_t Dim>
int dispatch(const std::string &execspace, int_t iterations,
             const std::string &func, const std::string &source_mesh,
             const std::string &target_mesh) {
  IterationResult res = std::visit(
      [&](auto execspace_t, auto rbf_function_t) {
        return _run_test<decltype(execspace_t), decltype(rbf_function_t), Dim>(
            source_mesh, target_mesh, iterations);
      },
      make_execspace(execspace), make_rbffunction(func));

  std::cout << res << std::endl;

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <execspace> <iterations> <function> <dims> <source.vtu> "
                 "<target.vtu>"
              << std::endl;
    std::cerr << "execspace: Serial, OpenMP, Threads, Cuda, HIP, "
                 "SYCL\nfunction: WendlandC0, WendlandC2, WendlandC4, "
                 "WendlandC6, WendlandC8\ndims: 1, 2, 3"
              << std::endl;
    return EXIT_FAILURE;
  }
  int_t dim = std::stoi(argv[4]);
  if (dim <= 0 || dim > 3) {
    std::cerr << "The provided dim (" << dim << ") must be between 1 and 3!"
              << std::endl;
    return EXIT_FAILURE;
  }
  index_t iterations = std::stoi(argv[2]);
  if (iterations <= 0) {
    std::cerr << "The provided iterations number (" << iterations
              << ") must be greater than 0!" << std::endl;
    return EXIT_FAILURE;
  }

  Kokkos::ScopeGuard guard;
  {
    std::string str_execspace = std::string(argv[1]);
    std::string str_function = std::string(argv[3]);
    std::string str_source = std::string(argv[5]);
    std::string str_target = std::string(argv[6]);
    switch (dim) {
    case 1:
      return dispatch<1>(str_execspace, iterations, str_function, str_source,
                         str_target);
    case 2:
      return dispatch<2>(str_execspace, iterations, str_function, str_source,
                         str_target);
    case 3:
      return dispatch<3>(str_execspace, iterations, str_function, str_source,
                         str_target);
    default:
      std::abort(); // Unreachable
    }
    return EXIT_SUCCESS;
  }
  return EXIT_SUCCESS;
}