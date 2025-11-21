#pragma once

#include <Eigen/Dense>
#include <utils/operators.hpp>

template <typename XType, typename XOffsType, typename RbfFuncType>
inline void HostFillMatrixA(Eigen::MatrixXd& A, const XType& X,
                            const XOffsType& XOffs, const RbfFuncType& func)
{
    const auto N = A.rows();
    const auto M = A.cols();
    for (auto i = static_cast<decltype(N)>(0); i < N; ++i)
    {
        for (auto j = i; j < M; ++j)
        {
            const auto val = func.eval(NDdistance(X(XOffs(i)), X(XOffs(j))));
            A(i, j) = val;
            A(j, i) = val;
        }
    }
}

template <typename XType, typename XOffsType>
inline void HostFillPoly(Eigen::MatrixXd& P, const XType& X,
                         const XOffsType& XOffs)
{
    const auto N = P.rows();
    const auto M = P.cols();

    for (auto i = static_cast<decltype(N)>(0); i < N; ++i)
    {
        const auto target = X(XOffs(i));
        P(i, 0) = 1.0;
        for (auto j = static_cast<decltype(M)>(1); j < M; ++j)
        {
            P(i, j) = target[j - 1];
        }
    }
}

template <typename XType, typename XOffsType>
inline void HostFillVec(Eigen::VectorXd& V, const XType& X,
                        const XOffsType& XOffs)
{
    const auto N = V.size();
    for (auto i = static_cast<decltype(N)>(0); i < N; ++i)
    {
        V(i) = X(XOffs(i));
    }
}

template <typename XType, typename XOffsType, typename YType,
          typename YOffsType, typename RbfFuncType>
inline void HostFillEvalMat(Eigen::MatrixXd& A, const XType& X,
                            const XOffsType& XOffs, const YType& Y,
                            const YOffsType& YOffs, const RbfFuncType& func)
{
    const auto N = A.rows();
    const auto M = A.cols();

    for (auto i = static_cast<decltype(N)>(0); i < N; ++i)
    {
        const auto target_point = Y(YOffs(i));
        for (auto j = static_cast<decltype(M)>(0); j < M; ++j)
        {
            const auto source_point = X(XOffs(j));
            A(i, j) = func.eval(NDdistance(target_point, source_point));
        }
    }
}
