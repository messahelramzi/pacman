#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "callbacks.hpp"
#include "interpolator.hxx"
#include "solver/device/kokkoskernels_matrices_op.hpp"
#include "solver/device/kokkoskernels_solver.hpp"
#include "solver/host/eigen_matrices_op.hpp"
#include "solver/host/eigen_solver.hpp"
#include "utils/utils.hpp"

FULL_TEMPLATE
constexpr void TEMPLATED_CLASSNAME::solve_systems(void)
{
    const std::string _region_name = "RbfPumInterpolator::solve_systems";
    Kokkos::Profiling::pushRegion(_region_name);

    if constexpr (is_host_accessible<ExecSpace>())
    {
        host_solve_systems();
    }
    else
    {
        device_solve_systems();
    }

    Kokkos::Profiling::popRegion();
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::device_solve_systems(void)
{
    const std::string _region_name = "RbfPumInterpolator::device_solve_systems";
    ExecSpace execspace{};

    constexpr double epsilon = 1.0e-14;

    // 1. On récupère les indices des points appartenant à chaque cluster
    const size_t K = this->_clusters.extent(0);

    VectorView<unsigned int> clusters_source_indices(
        _region_name + "::clusters_source_indices", 0);
    VectorView<int> clusters_source_offsets(
        _region_name + "::clusters_source_offsets", 0);
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_clusters_points_predicate(
        this->_clusters, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_clusters_points_callback;

    this->_source_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_source_indices, clusters_source_offsets);

    VectorView<unsigned int> clusters_target_indices(
        _region_name + "::clusters_target_indices", 0);
    VectorView<int> clusters_target_offsets(
        _region_name + "::clusters_target_offsets", 0);

    get_clusters_points_predicate.radius -= epsilon;
    this->_target_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_target_indices, clusters_target_offsets);

    // On cherche la taille des allocations pour les matrices A et b de chaque
    // cluster
    VectorView<int> rbf_mat_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::rbf_mat_offsets"),
        K + 1);
    VectorView<int> eval_mat_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::eval_mat_offsets"),
        K + 1);
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

    VectorView<ScalarType> As(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::As"),
                              rbf_mat_size);
    VectorView<ScalarType> Bs(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Bs"),
                              source_size);
    VectorView<ScalarType> Qs(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Qs"),
                              source_size * poly_vals);
    VectorView<ScalarType> Vs(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Vs"),
                              target_size * poly_vals);

    // 2. On remplit le système associé à chaque cluster
    // 2.1 On copie les données de this dont on a besoin
    const VectorView<Point> centers = this->_clusters;
    const VectorView<Point> source = this->_source;
    const VectorView<Point> target = this->_target;
    const VectorView<ScalarType> values = this->_values;
    const auto rbf_function = this->_rbf_function;
    const auto weighting_function = this->_weighting_function;

    const auto center_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(centers)
    };
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_target_clusters(
        target, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_target_clusters_callback;
    VectorView<unsigned int> weights_indices(_region_name + "::weights_indices",
                                             0);
    VectorView<int> weights_offsets(_region_name + "::weights_offsets", 0);
    center_bvh.query(execspace, get_target_clusters,
                     get_target_clusters_callback, weights_indices,
                     weights_offsets);

    VectorView<ScalarType> weights(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::weights"),
        weights_indices.extent(0));

    Kokkos::parallel_for(
        _region_name + "::p_for compute weights",
        Kokkos::RangePolicy(execspace, 0, target.extent(0)),
        KOKKOS_LAMBDA(const int& i) {
            const auto target_point = target(i);
            ScalarType sum_w = static_cast<ScalarType>(0);
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                const auto center_point = centers(weights_indices(k));
                const auto w = weighting_function.eval(
                    NDdistance(target_point, center_point));
                sum_w += w;
                weights(k) = w;
            }
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                weights(k) /= sum_w;
            }
        });
    Kokkos::fence();

    int max_cluster_size;
    Kokkos::parallel_reduce(
        _region_name + "::p_reduce find max cluster size",
        Kokkos::RangePolicy(execspace, 0, K),
        KOKKOS_LAMBDA(const int& k, int& lmax) {
            int s = clusters_source_offsets(k + 1) - clusters_source_offsets(k);
            lmax = (s > lmax) ? (s) : (lmax);
        },
        Kokkos::Max<int>(max_cluster_size));

    this->out = VectorView<ScalarType>(
        Kokkos::view_alloc(execspace, "this->out"), this->_target.extent(0));
    auto output = Kokkos::subview(this->out, Kokkos::ALL());

    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = team_policy::member_type;

    Kokkos::parallel_for(
        _region_name + "::p_for fill matrices",
        team_policy(ExecSpace{}, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            const auto k = team.league_rank();

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
            auto Q =
                GetPolyMatrix(k, Qs, clusters_source_offsets, n, poly_vals);
            auto V =
                GetPolyMatrix(k, Vs, clusters_target_offsets, m, poly_vals);
            auto B = GetRbfVector(k, Bs, clusters_source_offsets, n);

            TeamFillMatrixA(team, A, source, source_points, rbf_function);
            TeamFillPoly(team, Q, source, source_points);
            TeamFillPoly(team, V, target, target_points);
            TeamFillVec(team, B, values, source_points);
        });
    Kokkos::fence();

    VectorView<ScalarType> Ts(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Ts"),
                              poly_vals * K);
    VectorView<ScalarType> Ws(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Ws"),
                              source_size);
    VectorView<ScalarType> Xs(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Xs"),
                              source_size);

    // TODO: solve using Serial Solvers
    Kokkos::parallel_for(
        _region_name + "::p_for solve systems",
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

            auto Q =
                GetPolyMatrix(k, Qs, clusters_source_offsets, n, poly_vals);
            auto B = GetRbfMatrix(k, Bs, clusters_source_offsets, n, 1);
            auto X = GetRbfMatrix(k, Xs, clusters_source_offsets, n, 1);
            auto W = GetRbfVector(k, Ws, clusters_source_offsets, n);
            auto T = Kokkos::subview(
                Ts, Kokkos::make_pair(k * poly_vals, (k + 1) * poly_vals));

            SerialSolveQR(Q, B, T, X, W);

            SerialFillPoly(Q, source, source_points);
            const auto polynomial_contrib =
                Kokkos::subview(X, Kokkos::make_pair(0, poly_vals), 0);
            SerialMulMatVec(Q, polynomial_contrib, W);
            auto b = Kokkos::subview(B, Kokkos::ALL(), 0);
            SerialVecVecSub(b, W);
        });
    Kokkos::fence();

    VectorView<ScalarType> Es(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::Es"),
                              eval_mat_size);
    VectorView<ScalarType> tempView(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::tempView"),
        target_size);

    Kokkos::parallel_for(
        _region_name + "::p_for fill eval mat and build results",
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
            auto B = GetRbfMatrix(k, Bs, clusters_source_offsets, n, 1);
            auto X = GetRbfMatrix(k, Xs, clusters_source_offsets, n, 1);
            auto V =
                GetPolyMatrix(k, Vs, clusters_target_offsets, m, poly_vals);

            auto EvalMat = GetRbfMatrix(k, Es, eval_mat_offsets, m, n);
            TeamFillEvalMat(team, EvalMat, source, source_points, target,
                            target_points, rbf_function);
            auto tempOut =
                GetRbfVector(k, tempView, clusters_target_offsets, m);

            auto b = Kokkos::subview(B, Kokkos::ALL(), 0);
            auto x = Kokkos::subview(X, Kokkos::make_pair(0, poly_vals), 0);
            TeamSolveLU(team, A, b);
            team.team_barrier();

            TeamMulMatVec(team, EvalMat, b, tempOut);
            team.team_barrier();

            // TODO: peut etre un peu risqué
            auto EvalMatSlice = Kokkos::subview(EvalMat, Kokkos::ALL(), 0);
            TeamMulMatVec(team, V, x, EvalMatSlice);
            team.team_barrier();

            TeamVecVecAdd(team, tempOut, EvalMatSlice);
            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, m), [&](const int& i) {
                    const auto target_indice = target_points(i);
                    for (auto t = weights_offsets(target_indice);
                         t < weights_offsets(target_indice + 1); ++t)
                    {
                        if (weights_indices(t) == k)
                        {
                            Kokkos::atomic_add(&(output(target_points(i))),
                                               weights(t) * tempOut(i));
                        }
                    }
                });
        });
    Kokkos::fence();
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::host_solve_systems(void)
{
    const std::string _region_name = "RbfPumInterpolator::host_solve_systems";
    const ExecSpace execspace{};
    constexpr double epsilon = 1.0e-14;
    const size_t K = this->_clusters.extent(0);

    // 1. On récupère les indices des points appartenant à chaque cluster
    Kokkos::Profiling::pushRegion(_region_name
                                  + "::Get Source Points Per Cluster");
    VectorView<unsigned int> clusters_source_indices(
        _region_name + "::clusters_source_indices", 0);
    VectorView<int> clusters_source_offsets(
        _region_name + "::clusters_source_offsets", 0);
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_clusters_points_predicate(
        this->_clusters, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_clusters_points_callback;
    this->_source_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_source_indices, clusters_source_offsets);
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(_region_name
                                  + "::Get Target Points Per Cluster");
    VectorView<unsigned int> clusters_target_indices(
        _region_name + "::clusters_target_indices", 0);
    VectorView<int> clusters_target_offsets(
        _region_name + "::clusters_target_offsets", 0);
    get_clusters_points_predicate.radius -= epsilon;
    this->_target_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_target_indices, clusters_target_offsets);
    Kokkos::Profiling::popRegion();

    constexpr int poly_vals = Dim + 1;
    const auto source = this->_source;
    const auto target = this->_target;
    const auto values = this->_values;
    const auto centers = this->_clusters;
    const auto f = this->_rbf_function;
    const auto weighting_function = this->_weighting_function;

    Kokkos::Profiling::pushRegion(_region_name
                                  + "::Get Centers Per Target Point");
    const auto center_bvh = ArborX::BoundingVolumeHierarchy{
        execspace, ArborX::Experimental::attach_indices(centers)
    };
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_target_clusters(
        target, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_target_clusters_callback;
    VectorView<unsigned int> weights_indices(_region_name + "::weights_indices",
                                             0);
    VectorView<int> weights_offsets(_region_name + "::weights_offsets", 0);
    center_bvh.query(execspace, get_target_clusters,
                     get_target_clusters_callback, weights_indices,
                     weights_offsets);
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion(_region_name
                                  + "::Compute Weights Per Clusters");
    VectorView<ScalarType> weights(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::weights"),
        weights_indices.extent(0));
    Kokkos::parallel_for(
        _region_name + "::p_for compute weights",
        Kokkos::RangePolicy(execspace, 0, target.extent(0)), [=](const int& i) {
            const auto target_point = target(i);
            ScalarType sum_w = static_cast<ScalarType>(0.0);
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                const auto center_point = centers(weights_indices(k));
                const ScalarType w = weighting_function.eval_host(
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
    Kokkos::Profiling::popRegion();

    this->out = VectorView<ScalarType>(
        Kokkos::view_alloc(execspace, "this->out"), this->_target.extent(0));
    auto output = Kokkos::subview(this->out, Kokkos::ALL());

    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = team_policy::member_type;

    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                           Eigen::DontAlignCols, ",", "\n");

    Kokkos::parallel_for(
        _region_name + "::p_for solve systems",
        team_policy(execspace, K, Kokkos::AUTO), [=](const member_type& team) {
            const int k = team.league_rank();
            const int n =
                clusters_source_offsets(k + 1) - clusters_source_offsets(k);
            const int m =
                clusters_target_offsets(k + 1) - clusters_target_offsets(k);
            const auto source_sv = Kokkos::subview(
                clusters_source_indices,
                Kokkos::make_pair(clusters_source_offsets(k),
                                  clusters_source_offsets(k + 1)));
            const auto target_sv = Kokkos::subview(
                clusters_target_indices,
                Kokkos::make_pair(clusters_target_offsets(k),
                                  clusters_target_offsets(k + 1)));

            Eigen::MatrixXd A(n, n);
            TeamHostFillMatrixA(team, A, source, source_sv, f);
            // std::fstream file{ "./A_" + std::to_string(k) + ".csv",
            //                    file.binary | file.trunc | file.in | file.out
            //                    };
            // file << A.format(CSVFormat);

            Eigen::MatrixXd Q(n, poly_vals);
            TeamHostFillPoly(team, Q, source, source_sv);
            // std::fstream file2{ "./Q_" + std::to_string(k) + ".csv",
            //                     file.binary | file.trunc | file.in | file.out
            //                     };
            // file2 << Q.format(CSVFormat);

            Eigen::MatrixXd V(m, poly_vals);
            TeamHostFillPoly(team, V, target, target_sv);
            // std::fstream file3{ "./V_" + std::to_string(k) + ".csv",
            //                     file.binary | file.trunc | file.in | file.out
            //                     };
            // file3 << V.format(CSVFormat);

            Eigen::VectorXd B(n);
            TeamHostFillVec(team, B, values, source_sv);
            // std::fstream file4{ "./B_" + std::to_string(k) + ".csv",
            //                     file.binary | file.trunc | file.in | file.out
            //                     };
            // file4 << B.format(CSVFormat);

            Eigen::MatrixXd evalMat(m, n);
            TeamHostFillEvalMat(team, evalMat, source, source_sv, target,
                                target_sv, f);
            // std::fstream file5{ "./E_" + std::to_string(k) + ".csv",
            //                     file.binary | file.trunc | file.in | file.out
            //                     };
            // file5 << evalMat.format(CSVFormat);

            const auto polynomialContribution = HostSolveQR(Q, B);
            B -= (Q * polynomialContribution);

            const auto p = HostSolveLDLT(A, B);

            Eigen::VectorXd out = evalMat * p;

            out += (V * polynomialContribution);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, m), [&](const int& i) {
                    const auto target_indice = target_sv(i);
                    for (auto t = weights_offsets(target_indice);
                         t < weights_offsets(target_indice + 1); ++t)
                    {
                        if (static_cast<int>(weights_indices(t)) == k)
                        {
                            Kokkos::atomic_add<ScalarType>(
                                &(output(target_sv(i))), weights(t) * out(i));
                        }
                    }
                });
        });
    Kokkos::fence();
}
