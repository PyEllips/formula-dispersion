"""Some basic python tests"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from formula_dispersion import parse


def test_array_parsing():
    """Array parsing works properly"""

    parsed = parse("eps = 3 * 3 * lbda", "lbda", np.array([1.0, 2.0, 3.0]), {}, {})
    assert_array_almost_equal(parsed, np.array([9.0, 18.0, 27.0]))


def test_fail_2d_array():
    """Fails when passed a 2d array as wavelength axis"""
    with pytest.raises(TypeError):
        parse(
            "eps = 3 * 3 * lbda",
            "lbda",
            np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            {},
            {},
        )


def test_powc():
    """Using power is working properly"""

    parsed = parse("n = lbda ** 3", "lbda", np.array([1.0, 2.0, 3.0]), {}, {})
    assert_array_almost_equal(parsed, np.array([1.0**2, 8.0**2, 27.0**2]))


def test_sum():
    """The sum part works properly"""

    parsed = parse(
        "eps = 3 * sum[lbda * test]",
        "lbda",
        np.array([1.0, 2.0, 3.0]),
        {},
        {"test": [1.0, 2.0, 3.0]},
    )
    assert_array_almost_equal(
        parsed,
        3
        * (
            np.array([1.0, 2.0, 3.0])
            + 2 * np.array([1.0, 2.0, 3.0])
            + 3 * np.array([1.0, 2.0, 3.0])
        ),
    )


def test_fails_on_wrong_token():
    """Array parsing fails on wrong token"""

    with pytest.raises(TypeError):
        parse("eps = 3 * 3 * lba", "lbda", np.array([1.0, 2.0, 3.0]), {}, {})
