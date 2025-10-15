#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

#include "utils.hpp"

template <class ftype>
class RbfFunctionBasis
{
public:
    virtual ftype operator()(const ftype r) = 0;

private:
    Kokkos::View<ftype*> _params;
};

template <class ftype>
class WendlandC0 : public RbfFunctionBasis
{
public:
    ftype operator()(const ftype r)
    {
        // TODO: fill the evaluation formula
        return 0.;
    }

private:
    ftype _r_inv;
};

#endif /* ! RBF_FUNCTIONS_HPP */
