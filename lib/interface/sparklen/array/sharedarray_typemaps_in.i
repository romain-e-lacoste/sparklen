// Author : Romain E. Lacoste
// License : BSD-3-Clause

// NumPy arrays passed to the C++ core are wrapped without copying the data:
// each SharedArray is a temporary view on the NumPy buffer, marked as
// Python-owned, so C++ never frees it. The typemaps keep the arrays alive
// until the call returns, including sequences built on the fly. The arrays
// must be float64 in native byte order and C-contiguous, since the C++ side
// reads them as packed doubles.

////////////////////////////////////////////////
// Python NumPy Array ---> C++ SharedArray<T> //
////////////////////////////////////////////////

%typemap(in) (SharedArrayDouble1D &) (SharedArrayDouble1D res) {
	if (!PyArray_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "The input argument should be a NumPy Array");
        SWIG_fail;
    }
    if (PyArray_TYPE((PyArrayObject*)$input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "The data type of the NumPy Array should be double");
        SWIG_fail;
    }
	if (PyArray_NDIM((PyArrayObject*)$input) != 1) {
        PyErr_SetString(PyExc_ValueError, "The Expected Numpy Array should be 1-dimensional");
        SWIG_fail;
    }
    // The C++ side reads the buffer as packed doubles: reject strided views
    if (!PyArray_ISCARRAY_RO((PyArrayObject*)$input)) {
        PyErr_SetString(PyExc_ValueError, "The NumPy Array should be C-contiguous, aligned and in native byte order (use numpy.ascontiguousarray(x, dtype=numpy.float64))");
        SWIG_fail;
    }
    // Get NumPy array properties
    npy_intp size = PyArray_SIZE((PyArrayObject*)$input);
    double *data = static_cast<double*>(PyArray_DATA((PyArrayObject*)$input));

    // Create a SharedArray view of the NumPy buffer and mark that Python owns the data
    res = SharedArrayDouble1D(size, data);
    res.setPythonOwner(true);
    $1 = &res;
}


/////////////////////////////////////////////////////
// Python 2D NumPy Array ---> C++ SharedArray2D<T> //
/////////////////////////////////////////////////////

%typemap(in) (SharedArrayDouble2D &) (SharedArrayDouble2D res) {
    if (!PyArray_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "The input argument should be a NumPy Array");
        SWIG_fail;
    }
    if (PyArray_TYPE((PyArrayObject*)$input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "The data type of the NumPy Array should be double");
        SWIG_fail;
    }
	if (PyArray_NDIM((PyArrayObject*)$input) != 2) {
        PyErr_SetString(PyExc_ValueError, "The Expected Numpy Array should be 2-dimensional");
        SWIG_fail;
    }
    // The C++ side reads the buffer as row-major packed doubles: reject
    // Fortran-ordered arrays and strided views
    if (!PyArray_ISCARRAY_RO((PyArrayObject*)$input)) {
        PyErr_SetString(PyExc_ValueError, "The NumPy Array should be C-contiguous, aligned and in native byte order (use numpy.ascontiguousarray(x, dtype=numpy.float64))");
        SWIG_fail;
    }
    // Get NumPy array properties
    npy_intp n_rows = PyArray_DIM((PyArrayObject*)$input, 0);
    npy_intp n_cols = PyArray_DIM((PyArrayObject*)$input, 1);
    double *data = static_cast<double*>(PyArray_DATA((PyArrayObject*)$input));

    // Create a SharedArray2D view of the NumPy buffer and mark that Python owns the data
    res = SharedArrayDouble2D(n_rows, n_cols, data);
    res.setPythonOwner(true);
    $1 = &res;
}


//////////////////////////////////////////////////////////////////
// Python List of NumPy Array ---> C++ Vector of SharedArray<T> //
//////////////////////////////////////////////////////////////////

%typemap(in) (ListSharedArrayDouble1D &) (ListSharedArrayDouble1D res, PyObject *seq = NULL) {
    // Kept until the call returns (freearg): for a sequence other than a list
    // or a tuple, PySequence_Fast builds a new list, the only owner of its items
    seq = PySequence_Fast($input, "Expected a sequence");
    if (!seq) {
        SWIG_fail;  // PySequence_Fast has set TypeError
    }
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    res.resize(len);

    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyArray_Check(item)) {
            SWIG_exception_fail(SWIG_TypeError, "Expected a numpy array");
        }
        PyArrayObject *array = (PyArrayObject *)item;

        if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
            SWIG_exception_fail(SWIG_TypeError, "Expected a 1D numpy array of type double");
        }
        if (!PyArray_ISCARRAY_RO(array)) {
            SWIG_exception_fail(SWIG_ValueError, "The NumPy Array should be C-contiguous, aligned and in native byte order (use numpy.ascontiguousarray(x, dtype=numpy.float64))");
        }
        npy_intp size = PyArray_SIZE(array);
        double *data = static_cast<double *>(PyArray_DATA(array));

        res[i] = SharedArrayDouble1D(size, data);
        res[i].setPythonOwner(true);  // Mark that Python owns the data
    }
    $1 = &res;
}

%typemap(freearg) (ListSharedArrayDouble1D &) {
    // The views point into the arrays this sequence holds: release it only now
    Py_XDECREF(seq$argnum);
}


////////////////////////////////////////////////////////////////////////////////////
// Python List of List of NumPy Array ---> C++ Vector of Vector of SharedArray<T> //
////////////////////////////////////////////////////////////////////////////////////

%typemap(in) (ListListSharedArrayDouble1D &)
    (ListListSharedArrayDouble1D res, PyObject *seq_outer = NULL, PyObject *seqs_inner = NULL) {
    // Kept until the call returns (freearg): for a sequence other than a list
    // or a tuple, PySequence_Fast builds a new list, the only owner of its items
    seq_outer = PySequence_Fast($input, "Expected a sequence of sequences");
    if (!seq_outer) {
        SWIG_fail;  // PySequence_Fast has set TypeError
    }
    Py_ssize_t outer_len = PySequence_Fast_GET_SIZE(seq_outer);
    // Holds the inner sequences until the call returns
    seqs_inner = PyList_New(outer_len);
    if (!seqs_inner) {
        SWIG_fail;  // PyList_New has set MemoryError
    }
    res.resize(outer_len);

    for (Py_ssize_t i = 0; i < outer_len; ++i) {
        PyObject *seq_inner = PySequence_Fast(PySequence_Fast_GET_ITEM(seq_outer, i), "Expected a sequence");
        if (!seq_inner) {
            SWIG_fail;  // PySequence_Fast has set TypeError
        }
        PyList_SET_ITEM(seqs_inner, i, seq_inner);  // owned by seqs_inner from now on
        Py_ssize_t inner_len = PySequence_Fast_GET_SIZE(seq_inner);
        res[i].resize(inner_len);

        for (Py_ssize_t j = 0; j < inner_len; ++j) {
            PyObject *item_inner = PySequence_Fast_GET_ITEM(seq_inner, j);
            if (!PyArray_Check(item_inner)) {
                SWIG_exception_fail(SWIG_TypeError, "Expected a numpy array");
            }
            PyArrayObject *array = (PyArrayObject *)item_inner;

            if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
                SWIG_exception_fail(SWIG_TypeError, "Expected a 1D numpy array of type double");
            }
            if (!PyArray_ISCARRAY_RO(array)) {
                SWIG_exception_fail(SWIG_ValueError, "The NumPy Array should be C-contiguous, aligned and in native byte order (use numpy.ascontiguousarray(x, dtype=numpy.float64))");
            }
            npy_intp size = PyArray_SIZE(array);
            double *data = static_cast<double *>(PyArray_DATA(array));

            res[i][j] = SharedArrayDouble1D(size, data);
            res[i][j].setPythonOwner(true);  // Mark that Python owns the data
        }
    }
    $1 = &res;
}

%typemap(freearg) (ListListSharedArrayDouble1D &) {
    // The views point into the arrays these sequences hold: release them only now
    Py_XDECREF(seqs_inner$argnum);
    Py_XDECREF(seq_outer$argnum);
}
