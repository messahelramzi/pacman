#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

#include "utils.hpp"

template <class scalar_type>
class RbfFunctionBasis
{
public:
    virtual KOKKOS_INLINE_FUNCTION scalar_type
    operator()(const scalar_type r) const = 0;

private:
    Kokkos::View<scalar_type*> _params;
};

template <class scalar_type>
class WendlandC0 : public RbfFunctionBasis<scalar_type>
{
public:
    KOKKOS_INLINE_FUNCTION scalar_type operator()(const scalar_type r) const
    {
        const scalar_type p = r * _r_inv;
        if (p >= 1)
        {
            return (scalar_type)0.0;
        }
        return (1.0 - p) * (1.0 - p);
    }

private:
    scalar_type _r_inv;
};

template <class scalar_type>
class WendlandC2 : public RbfFunctionBasis<scalar_type>
{
public:
    KOKKOS_INLINE_FUNCTION scalar_type operator()(const scalar_type r) const
    {
        if (r < 0)
        {
            return (scalar_type)0.0;
        }
        const scalar_type p = r * _r_inv;
        if (p >= 1)
        {
            return (scalar_type)0.0;
        }
        return Kokkos::pow(1.0 - p, 4) * (4 * p + 1);
    }
    void set_r_inv(scalar_type r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    scalar_type _r_inv;
};

template <class scalar_type>
class WendlandC4 : public RbfFunctionBasis<scalar_type>
{
public:
    KOKKOS_INLINE_FUNCTION scalar_type operator()(const scalar_type r) const
    {
        const scalar_type p = r * _r_inv;
        if (p >= 1)
        {
            return (scalar_type)0.0;
        }
        return Kokkos::pow(1.0 - p, 6) * (35.0 * p * p + 18 * p + 3);
    }
    void set_r_inv(scalar_type r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    scalar_type _r_inv;
};

template <class scalar_type>
class WendlandC6 : public RbfFunctionBasis<scalar_type>
{
public:
    KOKKOS_INLINE_FUNCTION scalar_type operator()(const scalar_type r) const
    {
        const scalar_type p = r * _r_inv;
        if (p >= 1)
        {
            return (scalar_type)0.0;
        }
        return Kokkos::pow(1.0 - p, 8)
            * (32.0 * p * p * p + 25.0 * p * p + 8 * p + 1);
    }
    void set_r_inv(scalar_type r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    scalar_type _r_inv;
};

template <class scalar_type>
class WendlandC8 : public RbfFunctionBasis<scalar_type>
{
public:
    KOKKOS_INLINE_FUNCTION scalar_type operator()(const scalar_type r) const
    {
        const scalar_type p = r * _r_inv;
        if (p >= 1)
        {
            return (scalar_type)0.0;
        }
        return Kokkos::pow(1.0 - p, 10)
            * (1287.0 * p * p * p * p + 1350.0 * p * p * p + 630.0 * p * p
               + 150.0 * p + 15);
    }
    void set_r_inv(scalar_type r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    scalar_type _r_inv;
};

#endif /* ! RBF_FUNCTIONS_HPP */
