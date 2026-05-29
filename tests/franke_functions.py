#
# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
#

"""Shared Franke reference functions used by interpolation tests.

This module centralizes analytic reference surfaces used by both FE and RBF
functional tests so they stay consistent across test files.
"""

import numpy as np

def franke_2d(x, y):
    """Evaluate the 2D Franke function on NumPy arrays or scalars.

    Parameters
    ----------
    x, y : array_like
        Coordinates in the normalized domain.

    Returns
    -------
    numpy.ndarray or float
        Analytic reference values at ``(x, y)``.
    """
    return (
        0.75
        * np.exp(
            -(
                ((9.0 * x - 2.0) * (9.0 * x - 2.0))
                + ((9.0 * y - 2.0) * (9.0 * y - 2.0))
            )
            / 4.0
        )
        + 0.75
        * np.exp(
            -(
                ((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                + (9.0 * y + 1.0) / 10.0
            )
        )
        + 0.5
        * np.exp(
            -(
                ((9.0 * x - 7.0) * (9.0 * x - 7.0))
                + ((9.0 * y - 3.0) * (9.0 * y - 3.0))
            )
            / 4.0
        )
        - 0.2
        * np.exp(
            -(
                ((9.0 * x - 4.0) * (9.0 * x - 4.0))
                + ((9.0 * y - 7.0) * (9.0 * y - 7.0))
            )
        )
    )


def franke_3d(x, y, z):
    """Evaluate the 3D Franke-like extension on NumPy arrays or scalars.

    Parameters
    ----------
    x, y, z : array_like
        Coordinates in the normalized domain.

    Returns
    -------
    numpy.ndarray or float
        Analytic reference values at ``(x, y, z)``.
    """
    return (
        0.75
        * np.exp(
            -(
                ((9.0 * x - 2.0) * (9.0 * x - 2.0))
                + ((9.0 * y - 2.0) * (9.0 * y - 2.0))
                + ((9.0 * z - 2.0) * (9.0 * z - 2.0))
            )
            / 4.0
        )
        + 0.75
        * np.exp(
            -(
                ((9.0 * x + 1.0) * (9.0 * x + 1.0)) / 49.0
                + (9.0 * y + 1.0) / 10.0
                + (9.0 * z + 1.0) / 10.0
            )
        )
        + 0.5
        * np.exp(
            -(
                ((9.0 * x - 7.0) * (9.0 * x - 7.0))
                + ((9.0 * y - 3.0) * (9.0 * y - 3.0))
                + ((9.0 * z - 5.0) * (9.0 * z - 5.0))
            )
            / 4.0
        )
        - 0.2
        * np.exp(
            -(
                ((9.0 * x - 4.0) * (9.0 * x - 4.0))
                + ((9.0 * y - 7.0) * (9.0 * y - 7.0))
                + ((9.0 * z - 5.0) * (9.0 * z - 5.0))
            )
        )
    )