#ifndef RBF_FUNCTIONS_HPP
#define RBF_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>

#include "utils.hpp"

template <class ftype>
class RbfFunctionBasis {
    public:
        virtual ftype operator()(const ftype r) = 0;
    private:
        Kokkos::View<ftype*> _params;
};

template <class ftype>
class GaussianRbf : public RbfFunctionBasis<ftype> {
    public:
    ftype operator()(const ftype r) {
        return Kokkos::exp(-((r / this->_epsilon) * (r / this->_epsilon)));
    }

    private:
        ftype _epsilon = 1.0;
};

#endif /* ! RBF_FUNCTIONS_HPP */
