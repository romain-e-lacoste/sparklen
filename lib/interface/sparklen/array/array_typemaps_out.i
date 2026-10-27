// Author : Romain E. Lacoste
// License : BSD-3-Clause

//////////////////////////////////////////
// C++ Array<T> ---> Python Numpy Array //
//////////////////////////////////////////

// Typemap for creating back a Numpy Array from a C++ Array<double>
%typemap(out) ArrayDouble1D {
	// Retrieves the features of the C++ Array
    npy_intp dims[1] = { $1.size() };
    double *data = $1.data();

    // Create a NumPy array from the C++ Array features
    PyObject *obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*) data);
    if (!obj) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create back a NumPy array from C++ Array");
        SWIG_fail;
    }

    // Ensure the data is not deallocated when the Array object is destroyed
    PyArrayObject *py_array = (PyArrayObject*) obj;
    PyArray_CLEARFLAGS(py_array, NPY_ARRAY_OWNDATA);

    // Set a custom destructor to avoid double-free issues
    if (PyArray_SetBaseObject(py_array, PyCapsule_New((void*) data, NULL, NULL)) < 0) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "Unable to set base object");
        SWIG_fail;
    }
    $result = obj;
}


///////////////////////////////////////////////
// C++ Array2D<T> ---> Python 2D Numpy Array //
///////////////////////////////////////////////

// Typemap for creating back a 2D Numpy Array from a C++ Array2D<double>
%typemap(out) ArrayDouble2D {
	// Retrieves the features of the C++ Array
    npy_intp dims[2] = { $1.rows(), $1.cols() };
    double *data = $1.data();

    // Create a NumPy array from the C++ Array features
    PyObject *obj = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*) data);
    if (!obj) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to create back a 2D NumPy array from C++ Array2D");
        SWIG_fail;
    }

    // Ensure the data is not deallocated when the Array object is destroyed
    PyArrayObject *py_array = (PyArrayObject*) obj;
    PyArray_CLEARFLAGS(py_array, NPY_ARRAY_OWNDATA);

    // Set a custom destructor to avoid double-free issues
    if (PyArray_SetBaseObject(py_array, PyCapsule_New((void*) data, NULL, NULL)) < 0) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "Unable to set base object");
        SWIG_fail;
    }
    $result = obj;
}
