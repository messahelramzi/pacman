"""
Symbolic shape functions for VTK_PYRAMID (5-node pyramid) Lagrange element.

Reference element coordinates: (x, y, z) with z in [0, 1].
At z = 1 (apex): N4 = 1, N0..N3 = 0.
For z < 1:  N_i = f(x, y, z) / (4*(1 - z))
"""

import sympy as sp
import numpy as np

# ---------------------------------------------------------------------------
# Symbolic definitions
# ---------------------------------------------------------------------------
x, y, z = sp.symbols("x y z", real=True)

_denom = sp.Rational(1, 4) / (1 - z)

_x0 = -x + y + z - 1
_x1 = -x - y + z - 1
_x2 =  x - y + z - 1
_x3 =  x + y + z - 1

# Raw symbolic expressions (valid for z != 1)
_N_raw = [
    _x0 * _x1 * _denom,   # N0
    _x1 * _x2 * _denom,   # N1
    _x2 * _x3 * _denom,   # N2
    _x3 * _x0 * _denom,   # N3
    z,                     # N4 (apex)
]
_N_raw = [sp.simplify(n) for n in _N_raw]

VARS = (x, y, z)


def _gradient(expr):
    """Return list [dN/dx, dN/dy, dN/dz]."""
    return [sp.diff(expr, v) for v in VARS]


def _hessian(expr):
    """Return 3x3 sympy Matrix of second derivatives."""
    return sp.hessian(expr, VARS)


# Precompute gradients and Hessians symbolically
gradients = [_gradient(n) for n in _N_raw]
hessians  = [_hessian(n)  for n in _N_raw]

# ---------------------------------------------------------------------------
# Lambdified (numerical) callables
# ---------------------------------------------------------------------------
_shape_funcs_num = [sp.lambdify(VARS, n, modules="numpy") for n in _N_raw]
_grad_funcs_num  = [
    [sp.lambdify(VARS, g, modules="numpy") for g in grad]
    for grad in gradients
]
_hess_funcs_num  = [
    sp.lambdify(VARS, h, modules="numpy") for h in hessians
]

# Regularised limit values at z = 1 (apex)
# N0..N3 -> 0,  N4 = 1
# Gradients at apex taken from the C++ UpdateShapeFunctionsDerValues:
#   dN0/dx=0.5, dN0/dy=0.0, dN0/dz=-0.5
#   dN1/dx=0.0, dN1/dy=-0.5, dN1/dz=-0.5
#   dN2/dx=-0.5, dN2/dy=0.0, dN2/dz=-0.5
#   dN3/dx=0.0, dN3/dy=0.5, dN3/dz=-0.5
#   dN4/dx=0.0, dN4/dy=0.0, dN4/dz=1.0
_APEX_VALUES = [0.0, 0.0, 0.0, 0.0, 1.0]
_APEX_GRADS  = [
    [ 0.5,  0.0, -0.5],
    [ 0.0, -0.5, -0.5],
    [-0.5,  0.0, -0.5],
    [ 0.0,  0.5, -0.5],
    [ 0.0,  0.0,  1.0],
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_EPS = 1e-6


def shape_functions(xi: float, eta: float, phi: float) -> np.ndarray:
    """
    Evaluate all 5 pyramid shape functions at the reference point (xi, eta, phi).

    Parameters
    ----------
    xi, eta, phi : float
        Reference coordinates.  phi (z) must lie in [0, 1].

    Returns
    -------
    np.ndarray, shape (5,)
        Shape-function values [N0, N1, N2, N3, N4].
    """
    if abs(1.0 - phi) < _EPS:
        return np.array(_APEX_VALUES, dtype=float)
    return np.array([f(xi, eta, phi) for f in _shape_funcs_num], dtype=float)


def shape_function_gradients(xi: float, eta: float, phi: float) -> np.ndarray:
    """
    Evaluate the gradient of each shape function at (xi, eta, phi).

    Returns
    -------
    np.ndarray, shape (5, 3)
        grad[i, j] = dN_i / d{x,y,z}_j
    """
    if abs(1.0 - phi) < _EPS:
        return np.array(_APEX_GRADS, dtype=float)
    return np.array(
        [[g(xi, eta, phi) for g in row] for row in _grad_funcs_num],
        dtype=float,
    )


def shape_function_hessians(xi: float, eta: float, phi: float) -> np.ndarray:
    """
    Evaluate the Hessian of each shape function at (xi, eta, phi).

    Returns
    -------
    np.ndarray, shape (5, 3, 3)
        hess[i, j, k] = d²N_i / (d{x,y,z}_j  d{x,y,z}_k)
    """
    return np.array(
        [np.array(_hess_funcs_num[i](xi, eta, phi), dtype=float) for i in range(5)]
    )


# ---------------------------------------------------------------------------
# Pretty-print symbolic expressions
# ---------------------------------------------------------------------------
def print_symbolic():
    labels = ["N0", "N1", "N2", "N3", "N4"]
    dir_labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        print(f"\n{'='*60}")
        print(f"  {label} = {sp.simplify(_N_raw[i])}")
        print(f"  grad({label}) =")
        for j, d in enumerate(dir_labels):
            print(f"    d{label}/d{d} = {sp.simplify(gradients[i][j])}")
        print(f"  hessian({label}) =")
        sp.pprint(sp.simplify(hessians[i]))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_symbolic()

    print("\n" + "="*60)
    print("Partition-of-unity check at several points:")
    test_pts = [
        (0.0,  0.0, 0.0),
        (0.5,  0.2, 0.3),
        (-0.3, 0.4, 0.6),
        (0.0,  0.0, 1.0),   # apex
    ]
    for pt in test_pts:
        N = shape_functions(*pt)
        print(f"  (xi,eta,phi)={pt}  sum(N)={N.sum():.6f}  N={np.round(N, 6)}")

    print("\nGradient at (0.1, 0.2, 0.3):")
    G = shape_function_gradients(0.1, 0.2, 0.3)
    for i, row in enumerate(G):
        print(f"  grad N{i} = {np.round(row, 6)}")

    print("\nHessian of N0 at (0.1, 0.2, 0.3):")
    H = shape_function_hessians(0.1, 0.2, 0.3)
    print(np.round(H[0], 6))
