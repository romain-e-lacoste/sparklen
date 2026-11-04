# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Proximal operators work in place on the arrays they are given."""

import numpy as np
import pytest

from sparklen.prox import ProxL1


def test_prox_returns_x_and_rejects_arrays_it_cannot_update_in_place():
    """Check that apply updates x in place and returns it,
    and rejects arrays it cannot update in place."""
    prox = ProxL1()
    prox.set_pen_const(0.5)
    prox.set_application_range(1, 3)
    x = np.array([[1.0, 1.0, -1.0], [2.0, 0.2, 3.0]])
    assert prox.apply(x, 1.0) is x
    np.testing.assert_allclose(x, [[1.0, 0.5, 0.0], [2.0, 0.0, 2.5]])
    with pytest.raises(ValueError, match="C-contiguous"):
        prox.apply(np.asfortranarray(np.ones((2, 3))), 1.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        prox.apply(np.ones((2, 3), dtype=np.float32), 1.0)
    read_only = np.ones((2, 3))
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        prox.apply(read_only, 1.0)
