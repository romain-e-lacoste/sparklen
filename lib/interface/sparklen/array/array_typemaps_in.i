// Author : Romain E. Lacoste
// License : BSD-3-Clause

//////////////////////////////////////////
// Python Numpy Array ---> C++ Array<T> //
//////////////////////////////////////////

// Typemap for creating a C++ Array<double> from a Numpy Array
%typemap(in) (ArrayDouble1D &) (ArrayDouble1D res) {
    if (!PyArray_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "The input argument should be a NumPy Array");
        SWIG_fail;
    }
    if (PyArray_TYPE((PyArrayObject*)$input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "The data type of the NumPy Array should be double");
        SWIG_fail;
    }
	// Collects Numpy Array features
    npy_intp size = PyArray_SIZE((PyArrayObject*)$input);
    double *data = static_cast<double*>(PyArray_DATA((PyArrayObject*)$input));

	// Instantiation from the Numpy Array without taking ownership of the data
    res = ArrayDouble1D(size, data);
    $1 = &res;
}


///////////////////////////////////////////////
// Python 2D Numpy Array ---> C++ Array2D<T> //
///////////////////////////////////////////////

// Typemap for creating a C++ Array2D<double> from a 2D Numpy Array
%typemap(in) (ArrayDouble2D &) (ArrayDouble2D res) {
    if (!PyArray_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "The input argument should be a NumPy Array");
        SWIG_fail;
    }
    if (PyArray_TYPE($input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "The data type of the NumPy Array should be double");
        SWIG_fail;
    }
    if (PyArray_NDIM($input) != 2) {
        PyErr_SetString(PyExc_TypeError, "The Expected Numpy Array should be 2-dimensional");
        SWIG_fail;
    }

	// Collects Numpy Array features
    npy_intp *dims = PyArray_DIMS($input);
    npy_intp rows = dims[0];
    npy_intp cols = dims[1];
    double *data = static_cast<double*>(PyArray_DATA($input));

	// Instantiation from the 2D Numpy Array without taking ownership of the data
    res = ArrayDouble2D(rows, cols, data);
    $1 = &res;
}
