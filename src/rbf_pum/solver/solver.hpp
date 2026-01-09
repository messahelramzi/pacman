#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "common/concepts.hpp"
#include "common/types.hpp"
#include "interpolator.hxx"
#include "solver/utils/getters.hpp"
#include "solver/utils/serial_ops.hpp"
#include "solver/utils/serial_solve.hpp"
#include "solver/utils/team_ops.hpp"
#include "solver/utils/team_solve.hpp"
#include "utils/utils.hpp"

#define VIEW_ALLOC(name)                                                       \
  Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,                 \
                     _region_name + "::" + name)

namespace PACMAN {
namespace RbfPum {
FULL_TEMPLATE
void TEMPLATED_CLASSNAME::Interpolate(void) {
  const std::string _region_name = "RbfPumInterpolator::Interpolate";
  const ExecSpace execspace{};

  const index_t K = this->mClusters.extent(0);

  Kokkos::Profiling::pushRegion(_region_name +
                                "::fetch source and target nodes");
  VectorView<index_t> clusters_source_indices(
      VIEW_ALLOC("clusters_source_indices"), 0);
  VectorView<offset_t> clusters_source_offsets(
      VIEW_ALLOC("clusters_source_offsets"), 0);
  GetClustersPoints get_clusters_points_predicate{this->mClusters,
                                                  this->mRadius};
  GetClustersPointsCallback get_clusters_points_callback;
  VectorView<index_t> clusters_target_indices(
      VIEW_ALLOC("clusters_target_indices"), 0);
  VectorView<offset_t> clusters_target_offsets(
      VIEW_ALLOC("clusters_target_offsets"), 0);

  this->mSourceBvh.query(ExecSpace{}, get_clusters_points_predicate,
                         get_clusters_points_callback, clusters_source_indices,
                         clusters_source_offsets);
  get_clusters_points_predicate.radius -= fp_consts::epsilon();
  this->mTargetBvh.query(ExecSpace{}, get_clusters_points_predicate,
                         get_clusters_points_callback, clusters_target_indices,
                         clusters_target_offsets);
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // ! fetch source and target nodes

  Kokkos::Profiling::pushRegion(_region_name +
                                "::compute offsets and allocate memory");
  VectorView<offset_t> rbf_mat_offsets(VIEW_ALLOC("rbf_mat_offsets"), K + 1);
  VectorView<offset_t> eval_mat_offsets(VIEW_ALLOC("eval_mat_offsets"), K + 1);
  offset_t rbf_mat_size;
  offset_t eval_mat_size;
  OffsetsScan<ExecSpace> offsets_scan{clusters_source_offsets,
                                      clusters_target_offsets, rbf_mat_offsets,
                                      eval_mat_offsets};
  Kokkos::parallel_scan(_region_name + "::p_scan create access offsets",
                        Kokkos::RangePolicy(execspace, 0, K), offsets_scan);

  // TODO: we must avoid this
  auto sv1 = Kokkos::subview(rbf_mat_offsets, K);
  auto sv2 = Kokkos::subview(rbf_mat_offsets, 0);
  auto sv3 = Kokkos::subview(eval_mat_offsets, K);
  auto sv4 = Kokkos::subview(eval_mat_offsets, 0);
  Kokkos::deep_copy(rbf_mat_size, sv1);
  Kokkos::deep_copy(sv2, 0);
  Kokkos::deep_copy(eval_mat_size, sv3);
  Kokkos::deep_copy(sv4, 0);

  const int_t source_size = clusters_source_indices.extent_int(0);
  const int_t target_size = clusters_target_indices.extent_int(0);
  constexpr int_t poly_vals = Dim + 1;

  VectorView<fp_t> As(VIEW_ALLOC("As"), rbf_mat_size);
  VectorView<fp_t> Bs(VIEW_ALLOC("Bs"), source_size);
  VectorView<fp_t> Qs(VIEW_ALLOC("Qs"), source_size * poly_vals);
  VectorView<fp_t> Vs(VIEW_ALLOC("Vs"), target_size * poly_vals);
  VectorView<fp_t> Es(VIEW_ALLOC("Es"), eval_mat_size);
  VectorView<fp_t> Ts(VIEW_ALLOC("Ts"), poly_vals * K);
  VectorView<fp_t> Ws(VIEW_ALLOC("Ws"), source_size);
  VectorView<fp_t> Xs(VIEW_ALLOC("Xs"), source_size);
  VectorView<fp_t> TempView(VIEW_ALLOC("TempView"), target_size);
  VectorView<Kokkos::Array<bool, Dim>> AxisView(VIEW_ALLOC("AxisView"), K);

  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // ! compute offsets and allocate memory

  Kokkos::Profiling::pushRegion(_region_name + "::compute clusters weights");
  const auto centers = this->mClusters;
  const auto source = this->mSource;
  const auto target = this->mTarget;
  const auto values = this->mValues;
  const auto rbf_function = this->mRbfFunction;
  const auto weighting_function = this->mWeightingFunction;

  const auto center_bvh = ::ArborX::BoundingVolumeHierarchy{
      execspace, ::ArborX::Experimental::attach_indices(centers)};
  GetClustersPoints get_target_clusters{target, this->mRadius};
  GetClustersPointsCallback get_target_clusters_callback;
  VectorView<index_t> weights_indices(VIEW_ALLOC("weights_indices"), 0);
  VectorView<offset_t> weights_offsets(VIEW_ALLOC("weights_offsets"), 0);
  center_bvh.query(execspace, get_target_clusters, get_target_clusters_callback,
                   weights_indices, weights_offsets);

  VectorView<fp_t> weights(VIEW_ALLOC("weights"), weights_indices.extent(0));

  Kokkos::parallel_for(
      _region_name + "::p_for compute weights",
      Kokkos::RangePolicy(execspace, 0, target.extent(0)),
      KOKKOS_LAMBDA(const index_t &i) {
        const auto target_point = target(i);
        fp_t sum_w = fp_consts::zero();
        for (offset_t k = weights_offsets(i); k < weights_offsets(i + 1); ++k) {
          const auto center_point = centers(weights_indices(k));
          const auto w = weighting_function.Eval(
              DistanceNoCheck(target_point, center_point));
          sum_w += w;
          weights(k) = w;
        }
        for (offset_t k = weights_offsets(i); k < weights_offsets(i + 1); ++k) {
          weights(k) /= sum_w;
        }
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // ! compute clusters weights

  auto output = this->out;

  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = team_policy::member_type;

  Kokkos::Profiling::pushRegion(_region_name + "::fill and solve systems");
  Kokkos::parallel_for(
      _region_name + "::p_for team fill A, E, B",
      team_policy(execspace, K, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &team) {
        const index_t k = team.league_rank();
        const auto source_slice = Kokkos::make_pair(
            clusters_source_offsets(k), clusters_source_offsets(k + 1));
        const auto target_slice = Kokkos::make_pair(
            clusters_target_offsets(k), clusters_target_offsets(k + 1));

        const auto source_points =
            Kokkos::subview(clusters_source_indices, source_slice);
        const auto target_points =
            Kokkos::subview(clusters_target_indices, target_slice);

        const index_t n = source_slice.second - source_slice.first;
        const index_t m = target_slice.second - target_slice.first;

        auto A = GetRbfMatrix(k, As, rbf_mat_offsets, n, n);
        auto E = GetRbfMatrix(k, Es, eval_mat_offsets, m, n);
        auto B = GetRbfVector(k, Bs, clusters_source_offsets, n);
        TeamFillA(team, A, source, source_points, rbf_function);
        TeamFillE(team, E, source, source_points, target, target_points,
                  rbf_function);
        TeamFillB(team, B, values, source_points);
      });

  Kokkos::parallel_for(
      _region_name + "::p_for fill and solve remaining systems",
      Kokkos::RangePolicy(execspace, 0, K), KOKKOS_LAMBDA(const index_t &k) {
        const auto source_slice = Kokkos::make_pair(
            clusters_source_offsets(k), clusters_source_offsets(k + 1));
        const auto target_slice = Kokkos::make_pair(
            clusters_target_offsets(k), clusters_target_offsets(k + 1));

        const auto source_points =
            Kokkos::subview(clusters_source_indices, source_slice);
        const auto target_points =
            Kokkos::subview(clusters_target_indices, target_slice);

        const index_t n = source_slice.second - source_slice.first;
        const index_t m = target_slice.second - target_slice.first;

        auto A = GetRbfMatrix(k, As, rbf_mat_offsets, n, n);
        auto E = GetRbfMatrix(k, Es, eval_mat_offsets, m, n);
        auto B = GetRbfMatrix(k, Bs, clusters_source_offsets, n, 1);
        auto b = Kokkos::subview(B, Kokkos::ALL(), 0);

        auto active_axis = AxisView(k);
        for (int_t d = 0; d < active_axis.size(); ++d) {
          active_axis[d] = true;
        }
        auto tQ = GetPolyMatrix(k, Qs, clusters_source_offsets, n, active_axis);
        auto W = GetRbfVector(k, Ws, clusters_source_offsets, n);
        int_t pv = SerialFillQ(tQ, source, source_points, W, active_axis);

        auto V = GetPolyMatrix(k, Vs, clusters_target_offsets, m, active_axis);
        SerialFillPoly(V, target, target_points, active_axis);

        auto X = GetRbfMatrix(k, Xs, clusters_source_offsets, n, 1);
        auto T = Kokkos::subview(
            Ts, Kokkos::make_pair(k * poly_vals, k * poly_vals + pv));

        auto Q = GetPolyMatrix(k, Qs, clusters_source_offsets, n, active_axis);
        SerialSolveQR(Q, B, T, X, W);

        SerialFillPoly(Q, source, source_points, active_axis);
        const auto polynomial_contrib =
            Kokkos::subview(X, Kokkos::make_pair(0, pv), 0);
        SerialMulMatVec(Q, polynomial_contrib, W);

        SerialVecVecSub(b, W);

        auto temp_out = GetRbfVector(k, TempView, clusters_target_offsets, m);

        auto x = Kokkos::subview(X, Kokkos::make_pair(0, pv), 0);

        SerialSolveLU(A, b);
        SerialMulMatVec(E, b, temp_out);
        auto eval_mat_slice = Kokkos::subview(E, Kokkos::ALL(), 0);
        SerialMulMatVec(V, x, eval_mat_slice);
        SerialVecVecAdd(temp_out, eval_mat_slice);

        for (index_t i = 0; i < m; ++i) {
          const auto target_indice = target_points(i);
          const auto local_contribution = temp_out(i);
          const auto output_addr = &(output(target_indice));
          for (offset_t t = weights_offsets(target_indice);
               t < weights_offsets(target_indice + 1); ++t) {
            if (weights_indices(t) == k) {
              Kokkos::atomic_add(output_addr, weights(t) * local_contribution);
            }
          }
        }
      });
  Kokkos::fence();
  Kokkos::Profiling::popRegion(); // ! fill and solve systems
}
} // namespace RbfPum
} // namespace PACMAN
