import time
import pyomp
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit


def generate_data(n_features, n_samples, n_components):
    """
    Generate random dictionary and signal matrices for testing.
    """
    D = normalize(np.random.randn(n_features, n_components), axis=0)
    Y = np.random.randn(n_features, n_samples)
    return D, Y


def run_pyomp(D, Y, n_nonzero_coefs, eps=1e-6, gamma_mode=0, erroromp=0, DtX=None, XtX=None, G=None, profile=0, msg_delta=0):
    """
    Run the pyomp ompcore function and measure its performance.
    """
    gamma_mode = int(gamma_mode)
    erroromp = int(erroromp)
    profile = int(profile)

    t0 = time.time()
    X = pyomp.ompcore(
        D, Y, n_nonzero_coefs, eps, gamma_mode, erroromp,
        DtX=DtX, XtX=XtX, G=G, profile=profile, msg_delta=msg_delta
    )
    tf = time.time()
    print(f"pyomp.ompcore execution time: {tf - t0:.4f} seconds")
    return X, tf - t0


def run_sklearn_omp(D, Y, n_nonzero_coefs):
    """
    Run the sklearn OrthogonalMatchingPursuit and measure its performance.
    """
    t0 = time.time()
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False)
    omp.fit(D, Y)
    tf = time.time()
    print(f"sklearn.OMP execution time: {tf - t0:.4f} seconds")
    return omp.coef_.T, tf - t0


def compute_reconstruction_error(Y, D, X):
    """
    Compute and return the reconstruction error ||Y - DX||.
    """
    error = np.linalg.norm(Y - D @ X)
    return error


def main():
    # Parameters
    n_samples = 500
    n_features = 100
    n_components = 1000
    n_nonzero_coefs = 50
    eps = 1e-6

    # Generate data
    D, Y = generate_data(n_features, n_samples, n_components)

    # Run pyomp
    print("Running pyomp.ompcore...")
    X_pyomp, time_pyomp = run_pyomp(D, Y, n_nonzero_coefs, eps)

    # Compute and print pyomp reconstruction error
    error_pyomp = compute_reconstruction_error(Y, D, X_pyomp)
    print(f"pyomp reconstruction error: {error_pyomp:.6f}")

    # Run sklearn OMP
    print("Running sklearn OrthogonalMatchingPursuit...")
    X_sklearn, time_sklearn = run_sklearn_omp(D, Y, n_nonzero_coefs)

    # Compute and print sklearn reconstruction error
    error_sklearn = compute_reconstruction_error(Y, D, X_sklearn)
    print(f"sklearn OMP reconstruction error: {error_sklearn:.6f}")

    # Compare times
    print(f"pyomp is {time_sklearn / time_pyomp:.2f}x faster than sklearn OMP")

    # Compare reconstruction errors
    print(f"Reconstruction error difference: {abs(error_pyomp - error_sklearn):.6f}")


if __name__ == "__main__":
    main()
