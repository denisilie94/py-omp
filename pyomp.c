// pyomp.c

#include <Python.h>
#include <numpy/arrayobject.h>

#include "ompcore.h"
#include "omputils.h"
#include "ompprof.h"
#include "myblas.h"


static PyObject* py_ompcore(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *D_obj = NULL;
    PyObject *x_obj = NULL;
    PyObject *DtX_obj = Py_None;
    PyObject *XtX_obj = Py_None;
    PyObject *G_obj = Py_None;
    int T, gamma_mode, profile = 0, erroromp;
    double eps, msg_delta = 0.0;

    // Define keyword argument names
    static char *kwlist[] = {"D", "x", "T", "eps", "gamma_mode", "erroromp",
                             "DtX", "XtX", "G", "profile", "msg_delta", NULL};

    // Corrected format string
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOidii|OOOid", kwlist,
                                     &D_obj,
                                     &x_obj,
                                     &T,
                                     &eps,
                                     &gamma_mode,
                                     &erroromp,
                                     &DtX_obj,
                                     &XtX_obj,
                                     &G_obj,
                                     &profile,
                                     &msg_delta)) {
        return NULL;
    }

    // Import numpy array
    import_array();

    // Convert to NumPy arrays in Fortran order
    PyArrayObject *D_array = (PyArrayObject*) PyArray_FROMANY(D_obj, NPY_DOUBLE, 2, 2,
                                                              NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS);
    PyArrayObject *x_array = (PyArrayObject*) PyArray_FROMANY(x_obj, NPY_DOUBLE, 2, 2,
                                                              NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS);

    if (D_array == NULL || x_array == NULL) {
        Py_XDECREF(D_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    // Get dimensions
    npy_intp *D_dims = PyArray_DIMS(D_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    size_t n = (size_t)D_dims[0];
    size_t m = (size_t)D_dims[1];
    size_t L = (size_t)x_dims[1];

    // Handle optional arrays, ensuring they are in Fortran order if provided
    PyArrayObject *DtX_array = NULL;
    PyArrayObject *XtX_array = NULL;
    PyArrayObject *G_array = NULL;

    if (DtX_obj != Py_None) {
        DtX_array = (PyArrayObject*) PyArray_FROMANY(DtX_obj, NPY_DOUBLE, 2, 2,
                                                     NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS);
        if (DtX_array == NULL) {
            Py_DECREF(D_array);
            Py_DECREF(x_array);
            return NULL;
        }
    }
    if (XtX_obj != Py_None) {
        XtX_array = (PyArrayObject*) PyArray_FROMANY(XtX_obj, NPY_DOUBLE, 1, 1,
                                                     NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS);
        if (XtX_array == NULL) {
            Py_DECREF(D_array);
            Py_DECREF(x_array);
            Py_XDECREF(DtX_array);
            return NULL;
        }
    }
    if (G_obj != Py_None) {
        G_array = (PyArrayObject*) PyArray_FROMANY(G_obj, NPY_DOUBLE, 2, 2,
                                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS);
        if (G_array == NULL) {
            Py_DECREF(D_array);
            Py_DECREF(x_array);
            Py_XDECREF(DtX_array);
            Py_XDECREF(XtX_array);
            return NULL;
        }
    }

    // Get pointers to the data
    double *D = (double*) PyArray_DATA(D_array);
    double *x = (double*) PyArray_DATA(x_array);
    double *DtX = DtX_array ? (double*) PyArray_DATA(DtX_array) : NULL;
    double *XtX = XtX_array ? (double*) PyArray_DATA(XtX_array) : NULL;
    double *G = G_array ? (double*) PyArray_DATA(G_array) : NULL;

    // Prepare GammaMatrix structure
    GammaMatrix Gamma;
    memset(&Gamma, 0, sizeof(GammaMatrix));

    // Call ompcore
    ompcore(D, x, DtX, XtX, G, n, m, L,
            T, eps, gamma_mode, profile, msg_delta, erroromp, &Gamma);

    // Now convert Gamma to Python object to return
    PyObject *result = NULL;

    if (gamma_mode == 0) {
        // Gamma_full is a full matrix of size m x L (column-major in C)
        npy_intp dims[2] = { (npy_intp)m, (npy_intp)L };

        // Create the NumPy array without copying data
        PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
        PyObject *Gamma_full_array = PyArray_NewFromDescr(&PyArray_Type, descr, 2, dims,
                                                          NULL, Gamma.Gamma_full,
                                                          NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
        if (Gamma_full_array == NULL) {
            Py_DECREF(D_array);
            Py_DECREF(x_array);
            Py_XDECREF(DtX_array);
            Py_XDECREF(XtX_array);
            Py_XDECREF(G_array);
            return NULL;
        }

        // Transfer ownership of the data to Python
        PyArray_ENABLEFLAGS((PyArrayObject*)Gamma_full_array, NPY_ARRAY_OWNDATA);

        result = Gamma_full_array;
    } else {
        // Sparse matrix
        npy_intp nzmax = (npy_intp)(Gamma.gammaJc[Gamma.L]);
        PyObject *gammaPr_array = PyArray_SimpleNewFromData(1, &nzmax, NPY_DOUBLE, Gamma.gammaPr);
        PyObject *gammaIr_array = PyArray_SimpleNewFromData(1, &nzmax, NPY_INTP, Gamma.gammaIr);
        npy_intp gammaJc_size = (npy_intp)(Gamma.L + 1);
        PyObject *gammaJc_array = PyArray_SimpleNewFromData(1, &gammaJc_size, NPY_INTP, Gamma.gammaJc);

        PyArray_ENABLEFLAGS((PyArrayObject*)gammaPr_array, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS((PyArrayObject*)gammaIr_array, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS((PyArrayObject*)gammaJc_array, NPY_ARRAY_OWNDATA);

        result = Py_BuildValue("{s:O,s:O,s:O}",
                               "gammaPr", gammaPr_array,
                               "gammaIr", gammaIr_array,
                               "gammaJc", gammaJc_array);
    }

    // Clean up
    Py_DECREF(D_array);
    Py_DECREF(x_array);
    Py_XDECREF(DtX_array);
    Py_XDECREF(XtX_array);
    Py_XDECREF(G_array);

    return result;
}

/* Module method definition */
static PyMethodDef PyOMPCoreMethods[] = {
    {"ompcore", (PyCFunction)py_ompcore, METH_VARARGS | METH_KEYWORDS, "Execute the OMP algorithm."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef pyomp_module = {
    PyModuleDef_HEAD_INIT,
    "pyomp",
    NULL,
    -1,
    PyOMPCoreMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_pyomp(void) {
    PyObject *module;
    module = PyModule_Create(&pyomp_module);
    if (module == NULL)
        return NULL;
    import_array(); // Initialize NumPy API
    return module;
}
