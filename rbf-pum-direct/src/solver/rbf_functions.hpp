#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

template <class ScalarType>
class WendlandC0
{
public:
    KOKKOS_INLINE_FUNCTION ScalarType operator()(const ScalarType r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1)
        {
            return (ScalarType)0.0;
        }
        return (1.0 - p) * (1.0 - p);
    }

private:
    ScalarType _r_inv;
};

template <class ScalarType>
class WendlandC2
{
public:
    KOKKOS_FUNCTION ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        signed char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return (ScalarType)mask * Kokkos::pow(1.0 - p, 4)
            * Kokkos::fma(4, p, 1);
    }
    void set_r_inv(ScalarType r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ScalarType _r_inv;
};

template <class ScalarType>
class WendlandC4
{
public:
    KOKKOS_INLINE_FUNCTION ScalarType operator()(const ScalarType r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1)
        {
            return (ScalarType)0.0;
        }
        return Kokkos::pow(1.0 - p, 6) * (35.0 * p * p + 18 * p + 3);
    }
    void set_r_inv(ScalarType r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ScalarType _r_inv;
};

template <class ScalarType>
class WendlandC6
{
public:
    KOKKOS_INLINE_FUNCTION ScalarType operator()(const ScalarType r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1)
        {
            return (ScalarType)0.0;
        }
        return Kokkos::pow(1.0 - p, 8)
            * (32.0 * p * p * p + 25.0 * p * p + 8 * p + 1);
    }
    void set_r_inv(ScalarType r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ScalarType _r_inv;
};

template <class ScalarType>
class WendlandC8
{
public:
    KOKKOS_INLINE_FUNCTION ScalarType operator()(const ScalarType r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1)
        {
            return (ScalarType)0.0;
        }
        return Kokkos::pow(1.0 - p, 10)
            * (1287.0 * p * p * p * p + 1350.0 * p * p * p + 630.0 * p * p
               + 150.0 * p + 15);
    }
    void set_r_inv(ScalarType r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ScalarType _r_inv;
};

#endif /* ! RBF_FUNCTIONS_HPP */
