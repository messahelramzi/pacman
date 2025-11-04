#ifndef BATCHED_OPERATIONS_HPP
#define BATCHED_OPERATIONS_HPP

#include "interpolator.hxx"
#include <Kokkos_Core.hpp>

FULL_TEMPLATE
template <class BatchView, class OutView>
inline void TEMPLATED_CLASSNAME::batched_interpolate(BatchView &batch, OutView &out) const
{
    // TODO: replace mirror and copies by function arguments to copy once
    const std::string _region_name = "batched_interpolate";
    Kokkos::Profiling::pushRegion(_region_name);
    ExecSpace execspace{};
    const size_t N = this->_clusters.extent(0);
    const size_t K = batch.extent(0);
    Kokkos::View<Coordinates**, ExecSpace> weights(_region_name + "::weights",
                                                   K, N);
    Kokkos::View<bool**, ExecSpace> indices(_region_name + "::indices", K, N);

    auto clusters =
        Kokkos::create_mirror_view_and_copy(execspace, this->_clusters);
    auto weight_function = this->_weighting_function;

    Kokkos::View<Coordinates*, ExecSpace> weight_sums(
        _region_name + "::weight_sums", K);

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    Kokkos::parallel_for(
        _region_name
            + "::p_for compute intersecting clusters and their weights",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& member) {
            const size_t k = member.league_rank();
            Coordinates weight_sum;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, N),
                [&](const size_t& i, Coordinates& lsum) {
                    const Coordinates w =
                        weight_function(NDdistance(batch(k), clusters(i, 0)));
                    if (w > 0)
                    {
                        lsum += w;
                        weights(k, i) = w;
                        indices(k, i) = true;
                    }
                },
                weight_sum);
            Kokkos::single(Kokkos::PerTeam(member),
                           [=]() { weight_sums(k) = weight_sum; });
        });
    Kokkos::fence();

    Kokkos::View<Coordinates*, ExecSpace> interpolated_values(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing,
                           _region_name + "::interpolated values"),
        K);

    auto rbf_function = this->_rbf_function;
    auto coeffs = Kokkos::create_mirror_view_and_copy(execspace, this->_coeffs);
    auto bounds = Kokkos::create_mirror_view_and_copy(
        execspace, this->_nb_values_per_cluster);

    Kokkos::parallel_for(
        _region_name + "::p_for build global interpolant",
        team_policy(execspace, K, Kokkos::AUTO),
        KOKKOS_LAMBDA(const member_type& member) {
            const size_t k = member.league_rank();
            Coordinates interpolated_value;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, N),
                [&](const size_t& i, Coordinates& lsum) {
                    if (indices(k, i))
                    {
                        for (size_t ii = 0; ii < bounds(i); ++ii)
                        {
                            lsum += (weights(k, i) / weight_sums(k))
                                * (coeffs(i, ii)
                                   * (rbf_function(NDdistance(
                                       clusters(i, 1 + ii), batch(k)))));
                        }
                    }
                },
                interpolated_value);
            Kokkos::single(Kokkos::PerTeam(member), [=]() {
                interpolated_values(k) = interpolated_value;
            });
        });

    Kokkos::fence();

    Kokkos::deep_copy(out, interpolated_values);

    Kokkos::Profiling::popRegion(); // ! Interpolate
}

#endif // ! BATCHED_OPERATIONS_HPP