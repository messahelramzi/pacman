#pragma once

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

#include "utils/operators.hpp"
#include "utils/utils.hpp"

template <typename SourceType, typename OffsType, typename RbfFuncType>
Eigen::MatrixXd HostBuildSymmetricMatrix(const SourceType& src,
                                         const OffsType& offs,
                                         const RbfFuncType& func)
{
    const int n = offs.extent_int(0);
    Eigen::MatrixXd M(n, n);
    for (int i = 0; i < n; ++i)
    {
        const auto x = src(offs(i));
        for (int j = i; j < n; ++j)
        {
            const auto val =
                func.eval_host(NDdistance_no_check(x, src(offs(j))));
            M(i, j) = val;
            M(j, i) = val;
        }
    }
    return M;
}

template <typename SourceType, typename SourceOffsType, typename TargetType,
          typename TargetOffsType, typename RbfFuncType>
Eigen::MatrixXd
HostBuildRbfMatrix(const SourceType& src, const SourceOffsType& src_offs,
                   const TargetType& tgt, const TargetOffsType& tgt_offs,
                   const RbfFuncType& func)
{
    const int n = src_offs.extent_int(0);
    const int m = tgt_offs.extent_int(0);
    Eigen::MatrixXd M(m, n);
    for (int i = 0; i < m; ++i)
    {
        const auto x = tgt(tgt_offs(i));
        for (int j = 0; j < n; ++j)
        {
            M(i, j) = func.eval_host(NDdistance_no_check(x, src(src_offs(j))));
        }
    }
    return M;
}

template <typename ValuesType, typename OffsType>
Eigen::VectorXd HostBuildRbfVector(const ValuesType& values,
                                   const OffsType& offs)
{
    const int n = offs.extent_int(0);
    Eigen::VectorXd V(n);
    for (int i = 0; i < n; ++i)
    {
        V(i) = values(offs(i));
    }
    return V;
}

template <typename SourceType, typename OffsType, typename AxisType>
__forceinline__ int DeactivateOneAxis(const SourceType& src,
                                      const OffsType& offs,
                                      AxisType& active_axis)
{
    using ExecSpace = typename base_type<SourceType>::execution_space;
    constexpr int m = active_axis.size();
    int min_axis = 0;
    double min_distance = std::numeric_limits<double>::max();
    for (int i = 0; i < m; ++i)
    {
        if (!active_axis[i])
        {
            continue;
        }
        else
        {
            auto span = MyMinMax(offs, [&](const int& a, const int& b) {
                return src(a)[i] < src(b)[i];
            });
            const auto dist =
                Kokkos::abs(src(span.second)[i] - src(span.first)[i]);
            if (dist < min_distance)
            {
                min_distance = dist;
                min_axis = i;
            }
        }
    }
    active_axis[min_axis] = false;
    return min_axis;
}

template <typename SourceType, typename OffsType, typename AxisType>
Eigen::MatrixXd HostBuildSourcePoly(const SourceType& src, const OffsType& offs,
                                    AxisType& active_axis)
{
    const int n = offs.extent_int(0);
    constexpr int dim = active_axis.size();
    int poly_vals = dim + 1;
    for (int i = 0; i < dim; ++i)
    {
        active_axis[i] = true;
    }

    Eigen::MatrixXd P(n, poly_vals);
    do
    {
        P.conservativeResize(Eigen::NoChange, poly_vals);
        for (int i = 0; i < n; ++i)
        {
            const auto x = src(offs(i));
            P(i, 0) = 1.0;
            int k = 0;
            for (int j = 0; j < dim; ++j)
            {
                if (active_axis[j])
                {
                    P(i, 1 + k) = x[j];
                    ++k;
                }
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(P);
        const double condition = svd.singularValues()(0)
            / std::max(svd.singularValues()(svd.singularValues().size() - 1),
                       1.0e-14);
        if (condition > 1e5)
        {
            DeactivateOneAxis(src, offs, active_axis);
            --poly_vals;
        }
        else
        {
            break;
        }
    } while (true);
    return P;
}

template <typename TargetView, typename OffsView, typename AxisView>
Eigen::MatrixXd HostBuildTargetPoly(const TargetView& tgt, const OffsView& offs,
                                    const AxisView& active_axis)
{
    const int n = offs.extent_int(0);
    constexpr int dim = active_axis.size();
    int poly_vals = 1;
    for (int i = 0; i < dim; ++i)
    {
        if (active_axis[i])
        {
            ++poly_vals;
        }
    }
    Eigen::MatrixXd P(n, poly_vals);
    for (int i = 0; i < n; ++i)
    {
        const auto y = tgt(offs(i));
        P(i, 0) = 1.0;
        int k = 0;
        for (int j = 0; j < dim; ++j)
        {
            if (active_axis[j])
            {
                P(i, 1 + k) = y[j];
                ++k;
            }
        }
    }
    return P;
}
