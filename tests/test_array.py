"""Some basic python tests"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from formula_parser import parse_formula_dispersion

def test_array_parsing():
    """Array parsing works properly"""

    parsed = parse_formula_dispersion("3 * 3 * lbda", "lbda", np.array([1., 2., 3.]))
    assert_array_almost_equal(parsed, np.array([9., 18., 27.]))

def test_fails_on_wrong_token():
    """Array parsing fails on wrong token"""

    with pytest.raises(TypeError):
        parse_formula_dispersion("3 * 3 * lba", "lbda", np.array([1., 2., 3.]))
