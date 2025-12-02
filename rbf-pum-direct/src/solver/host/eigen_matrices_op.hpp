#pragma once

#include <Eigen/Dense>

#include "utils/operators.hpp"
#include "utils/utils.hpp"

template <typename TeamHandle, typename XType, typename XOffsType,
          typename RbfFuncType>
inline void TeamHostFillMatrixA(const TeamHandle& team_handle,
                                Eigen::MatrixXd& A, const XType& X,
                                const XOffsType& XOffs, const RbfFuncType& func)
{
    using data_type = typename base_type<decltype(A)>::Scalar;
    const auto N = static_cast<int>(A.rows());
    const auto M = static_cast<int>(A.cols());
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), [&](const int& i) {
            const auto source_point = X(XOffs(i));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_handle, i, M),
                                 [&](const int& j) {
                                     const data_type val = func.eval_host(
                                         NDdistance(source_point, X(XOffs(j))));
                                     A(i, j) = val;
                                     A(j, i) = val;
                                 });
        });
}

template <typename XType, typename XOffsType>
inline void HostFillPoly(Eigen::MatrixXd& P, const XType& X,
                         const XOffsType& XOffs)
{
    using data_type = typename base_type<decltype(P)>::Scalar;
    const data_type one = 1.0;
    const auto N = static_cast<int>(P.rows());
    const auto M = static_cast<int>(P.cols());
    for (auto i = 0; i < N; ++i)
    {
        const auto target = X(XOffs(i));
        P(i, 0) = one;
        for (auto j = 1; j < M; ++j)
        {
            P(i, j) = target[j - 1];
        }
    }
}

template <typename TeamHandle, typename XType, typename XOffsType>
inline void TeamHostFillPoly(const TeamHandle& team_handle, Eigen::MatrixXd& P,
                             const XType& X, const XOffsType& XOffs)
{
    using data_type = typename base_type<decltype(P)>::Scalar;
    const data_type one = static_cast<data_type>(1.0);
    const auto N = static_cast<int>(P.rows());
    const auto M = static_cast<int>(P.cols());
    Kokkos::parallel_for(Kokkos::TeamVectorMDRange(team_handle, N, M),
                         [&](const int& i, const int& j) {
                             char mask = static_cast<char>(j != 0);
                             P(i, j) =
                                 !mask * one + mask * X(XOffs(i))[j - mask * 1];
                         });
}

template <typename XType, typename XOffsType>
inline void HostFillVec(Eigen::VectorXd& V, const XType& X,
                        const XOffsType& XOffs)
{
    const auto N = static_cast<int>(V.size());
    for (auto i = 0; i < N; ++i)
    {
        V(i) = X(XOffs(i));
    }
}

template <typename TeamHandle, typename XType, typename XOffsType>
inline void TeamHostFillVec(const TeamHandle& team_handle, Eigen::VectorXd& V,
                            const XType& X, const XOffsType& XOffs)
{
    const auto N = static_cast<int>(V.size());
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_handle, N),
                         [&](const int& i) { V(i) = X(XOffs(i)); });
}

template <typename XType, typename XOffsType, typename YType,
          typename YOffsType, typename RbfFuncType>
inline void HostFillEvalMat(Eigen::MatrixXd& A, const XType& X,
                            const XOffsType& XOffs, const YType& Y,
                            const YOffsType& YOffs, const RbfFuncType& func)
{
    const auto N = static_cast<int>(A.rows());
    const auto M = static_cast<int>(A.cols());

    for (auto i = 0; i < N; ++i)
    {
        const auto target_point = Y(YOffs(i));
        for (auto j = 0; j < M; ++j)
        {
            A(i, j) =
                func.eval_host(NDdistance_no_check(target_point, X(XOffs(j))));
        }
    }
}

template <typename TeamHandle, typename XType, typename XOffsType,
          typename YType, typename YOffsType, typename RbfFuncType>
inline void TeamHostFillEvalMat(const TeamHandle& team_handle,
                                Eigen::MatrixXd& A, const XType& X,
                                const XOffsType& XOffs, const YType& Y,
                                const YOffsType& YOffs, const RbfFuncType& func)
{
    const auto N = static_cast<int>(A.rows());
    const auto M = static_cast<int>(A.cols());
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_handle, N), [&](const int& i) {
            const auto target_point = Y(YOffs(i));
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_handle, M), [&](const int& j) {
                    A(i, j) = func.eval_host(
                        NDdistance_no_check(target_point, X(XOffs(j))));
                });
        });
}
