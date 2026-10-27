// Author : Romain E. Lacoste
// License : BSD-3-Clause

////////////////////////////////////////////////
// Python Numpy Array ---> C++ SharedArray<T> //
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
	// Collects Numpy Array features
    npy_intp size = PyArray_SIZE((PyArrayObject*)$input);
    double *data = static_cast<double*>(PyArray_DATA((PyArrayObject*)$input));

	// Instantiation from the Numpy Array features and mark that Python owns the data
	res = SharedArrayDouble1D(size, data);
	res.setPythonOwner(true);
	$1 = &res;
}


/////////////////////////////////////////////////////
// Python 2D Numpy Array ---> C++ SharedArray2D<T> //
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
    // Collect NumPy Array features
    npy_intp n_rows = PyArray_DIM((PyArrayObject*)$input, 0);
    npy_intp n_cols = PyArray_DIM((PyArrayObject*)$input, 1);
    double *data = static_cast<double*>(PyArray_DATA((PyArrayObject*)$input));

    // Instantiate SharedArray2D from NumPy array features
    res = SharedArrayDouble2D(n_rows, n_cols, data);
    res.setPythonOwner(true);
    $1 = &res;
}


//////////////////////////////////////////////////////////////////
// Python List of Numpy Array ---> C++ Vector of SharedArray<T> //
//////////////////////////////////////////////////////////////////

%typemap(in) (ListSharedArrayDouble1D &) (ListSharedArrayDouble1D res) {
    PyObject *seq = PySequence_Fast($input, "Expected a sequence");
    if (!seq) {
        SWIG_exception_fail(SWIG_TypeError, "Expected a sequence");
    }
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    res.resize(len);

    for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyArray_Check(item)) {
            Py_DECREF(seq);
            SWIG_exception_fail(SWIG_TypeError, "Expected a numpy array");
        }
        PyArrayObject *array = (PyArrayObject *)item;

        if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
            Py_DECREF(seq);
            SWIG_exception_fail(SWIG_TypeError, "Expected a 1D numpy array of type double");
        }
        npy_intp size = PyArray_SIZE(array);
        double *data = static_cast<double *>(PyArray_DATA(array));

        res[i] = SharedArrayDouble1D(size, data);
        res[i].setPythonOwner(true);  // Mark that Python owns the data
    }
    Py_DECREF(seq);
    $1 = &res;
}


////////////////////////////////////////////////////////////////////////////////////
// Python List of List of Numpy Array ---> C++ Vector of Vector of SharedArray<T> //
////////////////////////////////////////////////////////////////////////////////////

%typemap(in) (ListListSharedArrayDouble1D &) (ListListSharedArrayDouble1D res) {
    PyObject *seq_outer = PySequence_Fast($input, "Expected a sequence of sequences");
    if (!seq_outer) {
        SWIG_exception_fail(SWIG_TypeError, "Expected a sequence of sequences");
    }
    Py_ssize_t outer_len = PySequence_Fast_GET_SIZE(seq_outer);
    res.resize(outer_len);

    for (Py_ssize_t i = 0; i < outer_len; ++i) {
        PyObject *item_outer = PySequence_Fast_GET_ITEM(seq_outer, i);
        PyObject *seq_inner = PySequence_Fast(item_outer, "Expected a sequence");
        if (!seq_inner) {
            Py_DECREF(seq_outer);
            SWIG_exception_fail(SWIG_TypeError, "Expected a sequence");
        }
        Py_ssize_t inner_len = PySequence_Fast_GET_SIZE(seq_inner);
        res[i].resize(inner_len);

        for (Py_ssize_t j = 0; j < inner_len; ++j) {
            PyObject *item_inner = PySequence_Fast_GET_ITEM(seq_inner, j);
            if (!PyArray_Check(item_inner)) {
                Py_DECREF(seq_inner);
                Py_DECREF(seq_outer);
                SWIG_exception_fail(SWIG_TypeError, "Expected a numpy array");
            }
            PyArrayObject *array = (PyArrayObject *)item_inner;

            if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
                Py_DECREF(seq_inner);
                Py_DECREF(seq_outer);
                SWIG_exception_fail(SWIG_TypeError, "Expected a 1D numpy array of type double");
            }
            npy_intp size = PyArray_SIZE(array);
            double *data = static_cast<double *>(PyArray_DATA(array));

            res[i][j] = SharedArrayDouble1D(size, data);
            res[i][j].setPythonOwner(true);  // Mark that Python owns the data
        }
        Py_DECREF(seq_inner);
    }
    Py_DECREF(seq_outer);
    $1 = &res;
}
