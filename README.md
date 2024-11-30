# pyomp: A Python and C Implementation of OMP

This package is inspired by Ron Rubinstein's **OMPbox**, originally implemented in MATLAB for solving sparse representation problems efficiently. The MATLAB implementation can be found at Ron Rubinstein's [official page](https://csaws.cs.technion.ac.il/~ronrubin/).

The goal of this package is to provide a high-performance implementation of Orthogonal Matching Pursuit (OMP) in Python and C, leveraging NumPy for Python bindings and direct C execution for maximum speed.

---

## Features

- **Python Implementation**: 
  - Provides an easy-to-use Python interface for calling the OMP algorithm.
  - Faster than the current OMP implementation in `scikit-learn` due to optimized logic and efficient use of NumPy arrays.
  - Example usage in `test_pyomp.py`.

- **C Implementation**: 
  - Offers a pure C implementation for maximum performance in environments where Python is not required.
  - Example usage in `test_ompcore.c`.

- **Inspired by OMPbox**: 
  - Follows the structure and ideas of Ron Rubinstein's MATLAB implementation of OMP, ensuring mathematical correctness and algorithmic rigor.

---

## Installation

To build and install the Python package with C extensions, use:

```bash
python setup.py build_ext --inplace
```

---

## Usage

### Python Implementation

Example usage for the Python implementation is provided in `test_pyomp.py`. Here's a quick overview:

```python
import numpy as np
from pyomp import ompcore

# Example dictionary and signals
D = np.random.rand(10, 20)
Y = np.random.rand(10, 5)

# Call ompcore with full control over parameters
X = ompcore(D, Y, T=5, eps=1e-6, gamma_mode=0, erroromp=0)
```

### C Implementation

For environments requiring raw speed and direct execution, use the C implementation. Compile and run the provided test:

```bash
gcc -o test_ompcore test_ompcore.c ompcore.c ompprof.c omputils.c myblas.c -lm
./test_ompcore
```

The test script `test_ompcore.c` demonstrates how to use the `ompcore` function with manually defined dictionaries and signals.

---

## Performance

The current Python implementation is faster than the `OMP` implementation available in `scikit-learn`. The performance boost is due to:

- Optimized use of NumPy's efficient array operations.
- Optional C implementation for further acceleration.

---

## TODO

- **`printspmat`**: The `printspmat` function is currently commented out. It will be revisited in future updates.
- **Support for Sparse Matrices**: Add native support for sparse dictionaries and signals to extend functionality for large-scale problems.

---

## Acknowledgments

- **Ron Rubinstein** for the original OMPbox MATLAB implementation that inspired this package.
- The open-source community for contributions and improvements.

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or report issues.

---

## License

This package is released under the MIT License. See the `LICENSE` file for details.