# Formula dispersion

A fast formula-parsing dispersion written in rust with python bindings.

# Installation

You can install it via pip:

```sh
pip install formula-dispersion
```

or as a development install from this git repository:

```sh
git clone https://github.com/PyEllips/formula-dispersion.git
cd formula-dispersion
pip install -e .
```

# Usage
This package provides one simple function `parse` you can use it like this

```python
from formula_dispersion import parse

parse(
        "eps = 1 + sum[A * (lbda * 1e-3)**2 / ((lbda * 1e-3)  ** 2 - B)]",
        "lbda",
        np.linspace(400, 1500, 500),
        {},
        {"A": [1, 1, 1], "B": [0.1, 0.1, 0.1]},
    )
```

which creates a sellmeier formula. The structure of parse is as follows:
```
parse(
    formula: The formula the be parsed
    wavelength idenitifier: The wavelength identifier to identify the wavelength array in the data
    wavelength axis: The axis points of the wavelength
    dictionary of single parameters:
      A dictionary of single parameters, i.e., ones which are not repeated in the formula
    dictionary of repeated parameters:
      A dictionary of repeated parameters, i.e., ones which are repeated in the `sum[...]` block of the formula.
)
```
