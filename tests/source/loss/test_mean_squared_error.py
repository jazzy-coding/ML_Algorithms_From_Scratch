"""Unit test suite for mean_squared_error module."""

import pytest
import numpy as np

from source.loss.mean_squared_error import mean_squared_error

# ------- TEST HAPPY PATH -------
def test_mean_squared_error():
    """Test mean_squared_error function with expected inputs."""
    # ARRANGE
    y = np.array([3., 5., 7.])
    y_hat = np.array([3., 5., 7.])

    # ACT
    result = mean_squared_error(y, y_hat)

    # ASSERT
    assert result == np.array([0.])
