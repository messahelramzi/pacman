#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "callbacks.hpp"
#include "interpolator.hxx"
#include "solver/utils/getters.hpp"
#include "solver/utils/serial_ops.hpp"
#include "solver/utils/serial_solve.hpp"
#include "solver/utils/team_ops.hpp"
#include "solver/utils/team_solve.hpp"
#include "utils/utils.hpp"

#define VIEW_ALLOC(name)                                                       \
    Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,               \
                       _region_name + "::" + name)

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::solve_systems(void)
{
    const std::string _region_name = "RbfPumInterpolator::solve_systems";
    const ExecSpace execspace{};

    constexpr double epsilon = 1.0e-14;

    const size_t K = this->_clusters.extent(0);

    Kokkos::Profiling::pushRegion(_region_name
                                  + "::fetch source and target nodes");
    VectorView<unsigned int> clusters_source_indices(
        VIEW_ALLOC("clusters_source_indices"), 0);
    VectorView<int> clusters_source_offsets(
        VIEW_ALLOC("clusters_source_offsets"), 0);
    GetClustersPoints get_clusters_points_predicate{ this->_clusters,
                                                     this->_radius };
    GetClustersPointsCallback get_clusters_points_callback;
    VectorView<unsigned int> clusters_target_indices(
        VIEW_ALLOC("clusters_target_indices"), 0);
    VectorView<int> clusters_target_offsets(
        VIEW_ALLOC("clusters_target_offsets"), 0);

    this->_source_bvh.query(ExecSpace{}, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_source_indices, clusters_source_offsets);
    get_clusters_points_predicate.radius -= epsilon;
    this->_target_bvh.query(ExecSpace{}, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_target_indices, clusters_target_offsets);
    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // ! fetch source and target nodes

    Kokkos::Profiling::pushRegion(_region_name
                                  + "::compute offsets and allocate memory");
    VectorView<int> rbf_mat_offsets(VIEW_ALLOC("rbf_mat_offsets"), K + 1);
    VectorView<int> eval_mat_offsets(VIEW_ALLOC("eval_mat_offsets"), K + 1);
    int rbf_mat_size;
    int eval_mat_size;
    OffsetsScan<ExecSpace> offsets_scan{ clusters_source_offsets,
                                         clusters_target_offsets,
                                         rbf_mat_offsets, eval_mat_offsets };
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

    const int source_size = clusters_source_indices.extent_int(0);
    const int target_size = clusters_target_indices.extent_int(0);
    constexpr int poly_vals = Dim + 1;

    VectorView<RbfPumFPType> As(VIEW_ALLOC("As"), rbf_mat_size);
    VectorView<RbfPumFPType> Bs(VIEW_ALLOC("Bs"), source_size);
    VectorView<RbfPumFPType> Qs(VIEW_ALLOC("Qs"), source_size * poly_vals);
    VectorView<RbfPumFPType> Vs(VIEW_ALLOC("Vs"), target_size * poly_vals);
    VectorView<RbfPumFPType> Es(VIEW_ALLOC("Es"), eval_mat_size);
    VectorView<RbfPumFPType> Ts(VIEW_ALLOC("Ts"), poly_vals * K);
    VectorView<RbfPumFPType> Ws(VIEW_ALLOC("Ws"), source_size);
    VectorView<RbfPumFPType> Xs(VIEW_ALLOC("Xs"), source_size);
    VectorView<RbfPumFPType> tempView(VIEW_ALLOC("tempView"), target_size);
    VectorView<Kokkos::Array<bool, Dim>> AxisView(VIEW_ALLOC("AxisView"), K);

    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // ! compute offsets and allocate memory

    Kokkos::Profiling::pushRegion(_region_name + "::compute clusters weights");
    const VectorView<Point> centers = this->_clusters;
    const VectorView<Point> source = this->_source;
    const VectorView<Point> target = this->_target;
    const VectorView<RbfPumFPType> values = this->_values;
    const auto rbf_function = this->_rbf_function;
    const auto weighting_function = this->_weighting_function;

    const auto center_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(centers)
    };
    GetClustersPoints get_target_clusters{ target, this->_radius };
    GetClustersPointsCallback get_target_clusters_callback;
    VectorView<unsigned int> weights_indices(VIEW_ALLOC("weights_indices"), 0);
    VectorView<int> weights_offsets(VIEW_ALLOC("weights_offsets"), 0);
    center_bvh.query(execspace, get_target_clusters,
                     get_target_clusters_callback, weights_indices,
                     weights_offsets);

    VectorView<RbfPumFPType> weights(VIEW_ALLOC("weights"),
                                     weights_indices.extent(0));

    Kokkos::parallel_for(
        _region_name + "::p_for compute weights",
        Kokkos::RangePolicy(execspace, 0, target.extent(0)),
        KOKKOS_LAMBDA(const int& i) {
            const auto target_point = target(i);
            RbfPumFPType sum_w = static_cast<RbfPumFPType>(0.0);
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                const auto center_point = centers(weights_indices(k));
                const auto w = weighting_function.eval(
                    NDdistance_no_check(target_point, center_point));
                sum_w += w;
                weights(k) = w;
            }
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                weights(k) /= sum_w;
            }
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // ! compute clusters weights

    this->out = VectorView<RbfPumFPType>(
        Kokkos::view_alloc(execspace, "this->out"), this->_target.extent(0));
    auto output = Kokkos::subview(this->out, Kokkos::ALL());

    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = team_policy::member_type;

    Kokkos::Profiling::pushRegion(_region_name + "::fill and solve systems");
    Kokkos::parallel_for(
        _region_name + "::p_for team fill A, E, B",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            const int k = team.league_rank();
            const auto source_slice = Kokkos::make_pair(
                clusters_source_offsets(k), clusters_source_offsets(k + 1));
            const auto target_slice = Kokkos::make_pair(
                clusters_target_offsets(k), clusters_target_offsets(k + 1));

            const auto source_points =
                Kokkos::subview(clusters_source_indices, source_slice);
            const auto target_points =
                Kokkos::subview(clusters_target_indices, target_slice);

            const auto n = source_slice.second - source_slice.first;
            const auto m = target_slice.second - target_slice.first;

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
        Kokkos::RangePolicy(execspace, 0, K), KOKKOS_LAMBDA(const int& k) {
            const auto source_slice = Kokkos::make_pair(
                clusters_source_offsets(k), clusters_source_offsets(k + 1));
            const auto target_slice = Kokkos::make_pair(
                clusters_target_offsets(k), clusters_target_offsets(k + 1));

            const auto source_points =
                Kokkos::subview(clusters_source_indices, source_slice);
            const auto target_points =
                Kokkos::subview(clusters_target_indices, target_slice);

            const auto n = source_slice.second - source_slice.first;
            const auto m = target_slice.second - target_slice.first;

            auto A = GetRbfMatrix(k, As, rbf_mat_offsets, n, n);
            auto E = GetRbfMatrix(k, Es, eval_mat_offsets, m, n);
            auto B = GetRbfMatrix(k, Bs, clusters_source_offsets, n, 1);
            auto b = Kokkos::subview(B, Kokkos::ALL(), 0);

            auto ActiveAxis = AxisView(k);
            for (int d = 0; d < ActiveAxis.size(); ++d)
            {
                ActiveAxis[d] = true;
            }
            auto tQ =
                GetPolyMatrix(k, Qs, clusters_source_offsets, n, ActiveAxis);
            auto W = GetRbfVector(k, Ws, clusters_source_offsets, n);
            int pv = SerialFillQ(tQ, source, source_points, W, ActiveAxis);

            auto V =
                GetPolyMatrix(k, Vs, clusters_target_offsets, m, ActiveAxis);
            SerialFillPoly(V, target, target_points, ActiveAxis);

            auto X = GetRbfMatrix(k, Xs, clusters_source_offsets, n, 1);
            auto T = Kokkos::subview(
                Ts, Kokkos::make_pair(k * poly_vals, k * poly_vals + pv));

            auto Q =
                GetPolyMatrix(k, Qs, clusters_source_offsets, n, ActiveAxis);
            SerialSolveQR(Q, B, T, X, W);

            SerialFillPoly(Q, source, source_points, ActiveAxis);
            const auto polynomial_contrib =
                Kokkos::subview(X, Kokkos::make_pair(0, pv), 0);
            SerialMulMatVec(Q, polynomial_contrib, W);

            SerialVecVecSub(b, W);

            auto tempOut =
                GetRbfVector(k, tempView, clusters_target_offsets, m);

            auto x = Kokkos::subview(X, Kokkos::make_pair(0, pv), 0);

            SerialSolveLU(A, b);
            SerialMulMatVec(E, b, tempOut);
            auto EvalMatSlice = Kokkos::subview(E, Kokkos::ALL(), 0);
            SerialMulMatVec(V, x, EvalMatSlice);
            SerialVecVecAdd(tempOut, EvalMatSlice);

            for (int i = 0; i < m; ++i)
            {
                const auto target_indice = target_points(i);
                const auto local_contribution = tempOut(i);
                const auto output_addr = &(output(target_indice));
                for (int t = weights_offsets(target_indice);
                     t < weights_offsets(target_indice + 1); ++t)
                {
                    if (weights_indices(t) == k)
                    {
                        Kokkos::atomic_add(output_addr,
                                           weights(t) * local_contribution);
                    }
                }
            }
        });
    Kokkos::fence();
    Kokkos::Profiling::popRegion(); // ! fill and solve systems
}