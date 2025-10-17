#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <ArborX_Point.hpp>
#include <Kokkos_Core.hpp>

#include "utils.hpp"

template <typename ExecSpace, int Dim, class ftype>
class Polynomial
{
public:
    virtual KOKKOS_INLINE_FUNCTION ArborX::Point<Dim + 1, ftype>
    operator()(const ArborX::Point<Dim, ftype>& point) const = 0;
};

template <typename ExecSpace, int Dim, class ftype>
class LinearPolynomial : public Polynomial<ExecSpace, Dim, ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ArborX::Point<Dim + 1, ftype>
    operator()(const ArborX::Point<Dim, ftype>& point) const
    {
        ArborX::Point<Dim + 1, ftype> out;
        out[0] = 1.0;
        for (int d = 0; d < Dim; ++d)
        {
            if (Kokkos::isnan(point[d]))
            {
                return ArborX::Point<Dim + 1, ftype>();
            }
            out[1 + d] = point[d];
        }
        return out;
    }

private:
    ExecSpace _execspace{};
};

#endif /* ! POLYNOMIAL_HPP */
