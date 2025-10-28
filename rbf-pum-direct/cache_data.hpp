#ifndef CACHE_DATA_HPP
#define CACHE_DATA_HPP

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <type_traits>

#include "interpolator.hxx"

FULL_TEMPLATE
void TEMPLATED_CLASSNAME::prepare_interpolation(void)
{
    assert(this->_clusters.extent(0) > 0 && this->_clusters.extent(1) > 0);
    const ExecSpace execspace{};
    const size_t N = _clusters.extent(1) - 1;
    const size_t M = _clusters.extent(0);
    const size_t Pn = this->_polynomial.get_base_size();
    Kokkos::View<Coordinates***, ExecSpace> all_lhs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_lhs"),
        M, N + Pn, N + Pn);
    Kokkos::View<Coordinates**, ExecSpace> all_rhs(
        Kokkos::view_alloc(
            execspace, Kokkos::WithoutInitializing,
            "RbfPumInterpolator::prepare_interpolation::all_rhs"),
        M, N + Pn);
    auto f = this->_rbf_function;
    auto clusters = Kokkos::create_mirror_view_and_copy(execspace, _clusters);
    auto values = Kokkos::create_mirror_view_and_copy(execspace, _values);
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_lhs rbf "
        "values (A)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU, 0x0LU },
                              { N, N, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            auto rbf_val = f(NDdistance<Dim, Coordinates>(clusters(k, i + 1),
                                                          clusters(k, j + 1)));
            all_lhs(k, i, j) = rbf_val;
            all_lhs(k, j, i) = rbf_val;
            all_rhs(k, i) = values(i);
        });
    auto p = this->_polynomial;
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs poly "
        "values (P/Pt)",
        Kokkos::MDRangePolicy(ExecSpace{}, { 0x0LU, 0x0LU }, { N, M }),
        KOKKOS_LAMBDA(const size_t& j, const size_t& k) {
            auto poly_values = p(clusters(k, 1 + j));
            for (size_t i = 0; i < Pn; ++i)
            {
                all_lhs(k, N + i, j) = poly_values[i];
                all_lhs(k, j, N + i) = poly_values[i];
            }
        });
    Kokkos::parallel_for(
        "RbfPumInterpolator::prepare_interpolation::p_for fill all_rhs padding "
        "zeros (0)",
        Kokkos::MDRangePolicy(execspace, { N, N, 0x0LU },
                              { N + Pn, N + Pn, M }),
        KOKKOS_LAMBDA(const size_t& i, const size_t& j, const size_t& k) {
            all_lhs(k, i, j) = 0.0;
            all_rhs(k, i) = 0.0;
        });
    Kokkos::fence();

    const size_t batch_size = M;
    const size_t mat_size = N + Pn;

    Kokkos::View<Coordinates**, Kokkos::LayoutRight, Kokkos::HostSpace> coeffs(
        Kokkos::view_alloc(Kokkos::HostSpace{}, Kokkos::WithoutInitializing,
                           "RbfPumInterpolator::prepare_interpolation::coeffs"),
        batch_size, mat_size);

    auto all_lhs_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all_lhs);
    auto all_rhs_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, all_rhs);

    constexpr auto Layout =
        (std::is_same_v<typename decltype(all_lhs_h)::array_layout,
                        Kokkos::LayoutLeft>)
        ? Eigen::ColMajor
        : Eigen::RowMajor;

    for (size_t i = 0; i < batch_size; ++i)
    {
        auto a_i = Kokkos::subview(all_lhs_h, i, Kokkos::ALL(), Kokkos::ALL());
        auto b_i = Kokkos::subview(all_rhs_h, i, Kokkos::ALL());
        auto lhs_i = Eigen::Map<
            Eigen::Matrix<Coordinates, Eigen::Dynamic, Eigen::Dynamic, Layout>>(
            a_i.data(), a_i.extent(0), a_i.extent(1));
        auto rhs_i = Eigen::Map<Eigen::Vector<Coordinates, Eigen::Dynamic>>(
            b_i.data(), b_i.extent(0));

        std::clog << "Matrix norm: " << lhs_i.norm() << "\n";

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            lhs_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd u = svd.solve(rhs_i);

        for (size_t ii = 0; ii < u.size(); ++ii)
        {
            coeffs(i, ii) = u[ii];
        }
    }
    this->_coeffs = Kokkos::View<Coordinates**, Kokkos::LayoutRight, ExecSpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           "this->_coeffs"),
        coeffs.extent(0), coeffs.extent(1));
    Kokkos::deep_copy(this->_coeffs, coeffs);
}

#endif /* ! CACHE_DATA_HPP */
