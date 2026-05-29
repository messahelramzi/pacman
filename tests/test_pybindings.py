#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

#   Linear mesh:                            Quadratic mesh:  
#
#         4                                        4
#        /\                                       /\
#       /  \            Field transfer        10 /  \9
#      /    \                                   /    \
#   3 /______\ 2        ------------->       3 /__ 7__\ 2  
#    |        |                               |        |
#    |        |                              8|        |6
#    |________|                               |________|
#   0          1                             0     5    1
#     
#

"""Functional FE interpolation test runner.

This script loads source/target meshes, evaluates python bindings,
calls ``pacman.fe.interpolate``, and validates results for multiple FE methods.
"""

import os
import sys
import pacman
import numpy as np
import argparse

def test_interpolation():
    """Run one FE interpolation validation case.
    """

    connVal = np.array([0, 1, 2, 3, 3, 2, 4], dtype=np.int64)
    connOff = np.array([0, 4, 7], dtype=np.int64)
    cellTypes = pacman.fe.vtk_to_pacman_cell_type(np.array([9, 5], dtype=np.int64)) # VTK_QUAD, VTK_TRI

    spaceDimension = 2

    sp = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 1.5]], dtype=np.float64)
    tp = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0.0], [1, 0.5], [0.5, 1], [0 ,0.5], [0.25, 1.25], [0.75, 1.25]], dtype=np.float64)
    print(tp.shape)
    sp_values = sp[:,0]

    tp_values, tp_status = pacman.fe.interpolate(spaceDimension, pacman.execspaces.OPENMP, pacman.fe.methods.INTERP_CLAMP, sp, sp_values, connVal, connOff, cellTypes, tp)

    tp_ref = tp[:,0]

    if not np.allclose(tp_values, tp_ref):
        print(f"tp_status: {tp_status}")
        print(f"tp_values: {tp_values}")
        print(f"tp_ref: {tp_ref}")
    assert np.allclose(tp_values, tp_ref), f"Method INTERP_CLAMP"

def main():
    test_interpolation()

if __name__ == "__main__":
    main()