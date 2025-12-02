#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

template <class ScalarType>
class WendlandC0
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<ScalarType>(mask)
            * ((static_cast<ScalarType>(1.0) - p)
               * (static_cast<ScalarType>(1.0) - p));
    }

private:
    ScalarType _r_inv;
};

template <class ScalarType>
class WendlandC2
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<ScalarType>(mask)
            * (Kokkos::pow(static_cast<ScalarType>(1.0) - p, 4)
               * Kokkos::fma(static_cast<ScalarType>(4.0), p,
                             static_cast<ScalarType>(1.0)));
    }
    ScalarType constexpr eval_host(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 4) * std::fma(4, p, 1);
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
    KOKKOS_FORCEINLINE_FUNCTION
    ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<ScalarType>(mask)
            * (Kokkos::pow(static_cast<ScalarType>(1.0) - p, 6)
               * (static_cast<ScalarType>(35.0) * Kokkos::pow(p, 2)
                  + Kokkos::fma(static_cast<ScalarType>(18.0), p,
                                static_cast<ScalarType>(3.0))));
    }
    ScalarType constexpr eval_host(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 6)
            * (35 * std::pow(p, 2) + std::fma(18, p, 3));
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
    KOKKOS_FORCEINLINE_FUNCTION
    ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<ScalarType>(mask)
            * (Kokkos::pow(static_cast<ScalarType>(1.0) - p, 8)
               * (static_cast<ScalarType>(32.0) * Kokkos::pow(p, 3)
                  + static_cast<ScalarType>(25.0) * Kokkos::pow(p, 2)
                  + Kokkos::fma(static_cast<ScalarType>(8.0), p,
                                static_cast<ScalarType>(1.0))));
    }
    ScalarType constexpr eval_host(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 8)
            * (32.0 * std::pow(p, 3) + 25.0 * std::pow(p, 2)
               + std::fma(8.0, p, 1.0));
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
    KOKKOS_FORCEINLINE_FUNCTION
    ScalarType constexpr eval(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        char mask = static_cast<char>(r >= 0.0 && p < 1.0 && p >= 0.0);
        return static_cast<ScalarType>(mask)
            * (Kokkos::pow(static_cast<ScalarType>(1.0) - p, 10)
                   * (static_cast<ScalarType>(1287.0) * Kokkos::pow(p, 4)
                      + static_cast<ScalarType>(1350.0) * Kokkos::pow(p, 3)
                      + static_cast<ScalarType>(630.0) * Kokkos::pow(p, 2))
               + Kokkos::fma(static_cast<ScalarType>(150.0), p,
                             static_cast<ScalarType>(15.0)));
    }
    ScalarType constexpr eval_host(const ScalarType& r) const
    {
        const ScalarType p = r * _r_inv;
        if (p >= 1 || p < 0)
        {
            return 0.0;
        }
        return std::pow(1.0 - p, 10)
            * (1287.0 * std::pow(p, 4) + 1350.0 * std::pow(p, 3)
               + 630.0 * std::pow(p, 2) + std::fma(150, p, 15));
    }
    void set_r_inv(ScalarType r_inv)
    {
        this->_r_inv = r_inv;
    }

private:
    ScalarType _r_inv;
};

#endif /* ! RBF_FUNCTIONS_HPP */
