#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

#include "utils.hpp"

template <class ftype>
class RbfFunctionBasis
{
public:
    virtual KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const = 0;

private:
    Kokkos::View<ftype*> _params;
};

template <class ftype>
class WendlandC0 : public RbfFunctionBasis<ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const
    {
        const ftype p = r * _r_inv;
        if (p >= 1 || Kokkos::isnan(r))
        {
            return (ftype)0.0;
        }
        return (1.0 - p) * (1.0 - p);
    }

private:
    ftype _r_inv;
};

template <class ftype>
class WendlandC2 : public RbfFunctionBasis<ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const
    {
        const ftype p = r * _r_inv;
        if (p >= 1 || Kokkos::isnan(r))
        {
            return (ftype)0.0;
        }
        return Kokkos::pow(1.0 - p, 4) * (4 * p + 1);
    }
    void set_r_inv(ftype r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ftype _r_inv;
};

template <class ftype>
class WendlandC4 : public RbfFunctionBasis<ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const
    {
        const ftype p = r * _r_inv;
        if (p >= 1 || Kokkos::isnan(r))
        {
            return (ftype)0.0;
        }
        return Kokkos::pow(1.0 - p, 6) * (35.0 * p * p + 18 * p + 3);
    }

private:
    ftype _r_inv;
};

template <class ftype>
class WendlandC6 : public RbfFunctionBasis<ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const
    {
        const ftype p = r * _r_inv;
        if (p >= 1 || Kokkos::isnan(r))
        {
            return (ftype)0.0;
        }
        return Kokkos::pow(1.0 - p, 8)
            * (32.0 * p * p * p + 25.0 * p * p + 8 * p + 1);
    }

private:
    ftype _r_inv;
};

template <class ftype>
class WendlandC8 : public RbfFunctionBasis<ftype>
{
public:
    KOKKOS_INLINE_FUNCTION ftype operator()(const ftype r) const
    {
        const ftype p = r * _r_inv;
        if (p >= 1 || Kokkos::isnan(r))
        {
            return (ftype)0.0;
        }
        return Kokkos::pow(1.0 - p, 10)
            * (1287.0 * p * p * p * p + 1350.0 * p * p * p + 630.0 * p * p
               + 150.0 * p + 15);
    }

private:
    ftype _r_inv;
};

#endif /* ! RBF_FUNCTIONS_HPP */
