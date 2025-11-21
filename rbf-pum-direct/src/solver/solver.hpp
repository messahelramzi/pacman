#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "callbacks.hpp"
#include "interpolator.hxx"
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

    constexpr double epsilon = 10e-14;

    // 1. On récupère les indices des points appartenant à chaque cluster
    const size_t K = this->_clusters.extent(0);

    VectorView<unsigned int> clusters_values_indices(
        _region_name + "::clusters_values_indices", 0);
    VectorView<int> clusters_values_offsets(
        _region_name + "::clusters_values_offsets", 0);
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_clusters_points_predicate(
        this->_clusters, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_clusters_points_callback;

    this->_source_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_values_indices, clusters_values_offsets);

    VectorView<unsigned int> target_values_indices(
        _region_name + "::target_values_indices", 0);
    VectorView<int> target_values_offsets(
        _region_name + "::target_values_offsets", 0);

    get_clusters_points_predicate.radius -= epsilon;
    this->_target_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback, target_values_indices,
                            target_values_offsets);

    // On cherche la taille des allocations pour les matrices A et b de chaque
    // cluster
    VectorView<int> As_access_offsets(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::access_offsets"),
        K + 1);
    int As_size;
    int bs_size;
    Kokkos::parallel_scan(
        _region_name + "::p_scan create access offsets for As",
        Kokkos::RangePolicy(execspace, 0, K),
        KOKKOS_LAMBDA(const int& i, int& update, const bool final) {
            update +=
                (clusters_values_offsets(i + 1) - clusters_values_offsets(i))
                * (clusters_values_offsets(i + 1) - clusters_values_offsets(i));
            if (final)
            {
                As_access_offsets(i + 1) = update;
            }
        });

    auto sv1 = Kokkos::subview(As_access_offsets, K);
    Kokkos::deep_copy(As_size, sv1);
    auto sv2 = Kokkos::subview(clusters_values_offsets, K);
    Kokkos::deep_copy(bs_size, sv2);
    auto sv3 = Kokkos::subview(As_access_offsets, 0);
    Kokkos::deep_copy(sv3, 0);

    VectorView<ScalarType> As(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::As"),
                              As_size);
    VectorView<ScalarType> bs(Kokkos::view_alloc(execspace,
                                                 Kokkos::WithoutInitializing,
                                                 _region_name + "::bs"),
                              bs_size);
    VectorView<ScalarType> Ps_in(Kokkos::view_alloc(execspace,
                                                    Kokkos::WithoutInitializing,
                                                    _region_name + "::Ps_in"),
                                 bs_size * (Dim + 1));
    VectorView<ScalarType> Ps_out(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::Ps_out"),
        target_values_indices.extent(0) * (Dim + 1));
    VectorView<ScalarType> coeffs(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::coeffs"),
        bs_size);

    // 2. On remplit le système associé à chaque cluster
    // 2.1 On copie les données de this dont on a besoin
    const VectorView<Point> centers = this->_clusters;
    const VectorView<Point> source = this->_source;
    const VectorView<Point> target = this->_target;
    const VectorView<ScalarType> values = this->_values;

    auto rbf_function = this->_rbf_function;
    constexpr int stride = Dim + 1;

    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = team_policy::member_type;

    Kokkos::parallel_for(
        _region_name + "::p_for fill rbf matrices",
        team_policy(ExecSpace{}, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            const size_t k = team.league_rank();
            const int access_index = As_access_offsets(k);
            const int next_index = As_access_offsets(k + 1);
            const int n =
                clusters_values_offsets(k + 1) - clusters_values_offsets(k);
            const auto range = Kokkos::make_pair(access_index, next_index);
            auto matrixCLU = Kokkos::subview(As, range);
            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange(team, n, n),
                [&](const int& i, const int& j) {
                    if (j >= i)
                    {
                        const auto p1 = source(clusters_values_indices(
                            clusters_values_offsets(k) + i));
                        const auto p2 = source(clusters_values_indices(
                            clusters_values_offsets(k) + j));
                        ScalarType r = NDdistance(p1, p2);
                        r = rbf_function.eval(r);
                        matrixCLU(i * n + j) = r;
                        matrixCLU(j * n + i) = r;
                    }
                });
        });
    Kokkos::parallel_for(
        _region_name + "::p_for fill bs vectors",
        team_policy(ExecSpace{}, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            const size_t k = team.league_rank();
            const int access_index = clusters_values_offsets(k);
            const int next_index = clusters_values_offsets(k + 1);
            const int n = next_index - access_index;
            const auto range = Kokkos::make_pair(access_index, next_index);
            auto b = Kokkos::subview(bs, range);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n),
                                 [&](const int& i) {
                                     b(i) = values(clusters_values_indices(
                                         clusters_values_offsets(k) + i));
                                 });
        });
    Kokkos::parallel_for(
        _region_name + "::p_for fill input polynomial matrices",
        team_policy(ExecSpace{}, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            TeamFillPolynomial(team, Ps_in, clusters_values_indices,
                               clusters_values_offsets, source, stride);
        });
    Kokkos::parallel_for(
        _region_name + "::p_for fill output polynomial matrices",
        team_policy(ExecSpace{}, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            TeamFillPolynomial(team, Ps_out, target_values_indices,
                               target_values_offsets, target, stride);
        });
    Kokkos::fence();

    // VectorView<ScalarType> Ws(Kokkos::view_alloc(execspace,
    //                                              Kokkos::WithoutInitializing,
    //                                              _region_name + "::Ws"),
    //                           bs_size);
    // VectorView<ScalarType> Ys(Kokkos::view_alloc(execspace,
    //                                              Kokkos::WithoutInitializing,
    //                                              _region_name + "::Ys"),
    //                           bs_size);
    // VectorView<ScalarType> Ts(Kokkos::view_alloc(execspace,
    //                                              Kokkos::WithoutInitializing,
    //                                              _region_name + "::Ts"),
    //                           bs_size);

    const auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
        _region_name + "::p_for solve systems",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& team) {
            const int k = team.league_rank();
            // A => matrixCLU
            auto A = TeamGetSystemA(team, As, As_access_offsets,
                                    clusters_values_offsets);
            // b => inputData
            auto b = TeamGetSystemB(team, bs, clusters_values_offsets);
            // P_in => matrixQ
            // auto P_in =
            //     TeamGetSystemP(team, Ps_in, clusters_values_offsets, stride);
            // // P_out => matrixV
            // auto P_out =
            //     TeamGetSystemP(team, Ps_out, target_values_offsets, stride);
            // auto w = TeamGetSystemW(team, Ws, clusters_values_offsets);
            // auto y = TeamGetSystemB(team, Ys, clusters_values_offsets);
            // auto t = TeamGetSystemB(team, Ts, clusters_values_offsets);

            // TeamSolveQR(team, P_in, b, y, t, w);
            // team.team_barrier();

            // Kokkos::single(Kokkos::PerTeam(team), [&](void) {
            //     Kokkos::printf("%lu. y = {%.12f, %.12f, %.12f, %.12f}\n", k,
            //                    y(0), y(1), y(2), y(3));
            // });

            TeamSolveLU(team, A, b);
            team.team_barrier();

            auto out = Kokkos::subview(
                coeffs,
                Kokkos::make_pair(
                    clusters_values_offsets(team.league_rank()),
                    clusters_values_offsets(team.league_rank() + 1)));
            KokkosBatched::TeamVectorCopy<
                member_type, KokkosBatched::Trans::NoTranspose, 1>::invoke(team,
                                                                           b,
                                                                           out);
        });
    Kokkos::fence();
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "solve time: " << (t2 - t1).count() / 1'000'000.0 << "ms"
              << std::endl;

    this->_coeffs = decltype(coeffs)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_coeffs"),
        coeffs.extent(0));
    Kokkos::deep_copy(this->_coeffs, coeffs);
}

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::host_solve_systems(void)
{
    const std::string _region_name = "RbfPumInterpolator::host_solve_systems";
    const ExecSpace execspace{};
    constexpr double epsilon = 10e-14;

    // 1. On récupère les indices des points appartenant à chaque cluster
    const size_t K = this->_clusters.extent(0);

    VectorView<unsigned int> clusters_values_indices(
        _region_name + "::clusters_values_indices", 0);
    VectorView<int> clusters_values_offsets(
        _region_name + "::clusters_values_offsets", 0);
    GetClustersPoints<ExecSpace, Dim, ScalarType> get_clusters_points_predicate(
        this->_clusters, this->_radius);
    GetClustersPointsCallback<ExecSpace, Dim, ScalarType>
        get_clusters_points_callback;

    this->_source_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback,
                            clusters_values_indices, clusters_values_offsets);

    VectorView<unsigned int> target_values_indices(
        _region_name + "::target_values_indices", 0);
    VectorView<int> target_values_offsets(
        _region_name + "::target_values_offsets", 0);

    get_clusters_points_predicate.radius -= epsilon;
    this->_target_bvh.query(execspace, get_clusters_points_predicate,
                            get_clusters_points_callback, target_values_indices,
                            target_values_offsets);

    constexpr int poly_vals = Dim + 1;
    const auto source = this->_source;
    const auto target = this->_target;
    const auto values = this->_values;
    const auto centers = this->_clusters;
    const auto f = this->_rbf_function;

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
        Kokkos::RangePolicy(execspace, 0, target.extent(0)), [=](const int& i) {
            const auto target_point = target(i);
            ScalarType sum_w = static_cast<ScalarType>(0);
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                const auto center_point = centers(weights_indices(k));
                const auto w = f.eval(NDdistance(target_point, center_point));
                sum_w += w;
                weights(k) = w;
            }
            for (int k = weights_offsets(i); k < weights_offsets(i + 1); ++k)
            {
                weights(k) /= sum_w;
            }
        });

    VectorView<ScalarType> output(
        Kokkos::view_alloc(execspace, _region_name + "::output"),
        this->_target.extent(0));
    Kokkos::parallel_for(
        _region_name + "p_for solve systems",
        Kokkos::RangePolicy(execspace, 0, K), [=](const int& k) {
            const int n =
                clusters_values_offsets(k + 1) - clusters_values_offsets(k);
            const int m =
                target_values_offsets(k + 1) - target_values_offsets(k);
            const auto source_sv = Kokkos::subview(
                clusters_values_indices,
                Kokkos::make_pair(clusters_values_offsets(k),
                                  clusters_values_offsets(k + 1)));
            const auto target_sv = Kokkos::subview(
                target_values_indices,
                Kokkos::make_pair(target_values_offsets(k),
                                  target_values_offsets(k + 1)));
            Eigen::MatrixXd A(n, n);
            HostFillMatrixA(A, source, source_sv, f);

            Eigen::MatrixXd Q(n, poly_vals);
            HostFillPoly(Q, source, source_sv);

            Eigen::MatrixXd V(m, poly_vals);
            HostFillPoly(V, target, target_sv);

            Eigen::VectorXd B(n);
            HostFillVec(B, values, source_sv);

            const auto polynomialContribution = HostSolveQR(Q, B);
            B -= (Q * polynomialContribution);

            const auto p = HostSolveLDLT(A, B);

            Eigen::MatrixXd evalMat(m, n);
            HostFillEvalMat(evalMat, source, source_sv, target, target_sv, f);

            Eigen::VectorXd out = evalMat * p;

            out += (V * polynomialContribution);

            for (auto i = 0; i < m; ++i)
            {
                const auto target_indice = target_sv(i);
                for (auto t = weights_offsets(target_indice);
                     t < weights_offsets(target_indice + 1); ++t)
                {
                    if (weights_indices(t) == k)
                    {
                        output(target_sv(i)) += weights(t) * out(i);
                    }
                }
            }
        });
    Kokkos::fence();

    this->out = decltype(output)(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "this->out"),
        output.extent(0));
    Kokkos::deep_copy(this->out, output);
}
