// Author : Romain E. Lacoste
// License : BSD-3-Clause

////////////////////////////////////////////////
// C++ SharedArray<T> ---> Python Numpy Array //
////////////////////////////////////////////////

%typemap(out) SharedArrayDouble1D {
    SharedArrayDouble1D arr = $1;

    // Retrieve size and data directly from the SharedArrayDouble1D
    size_t size = arr.size();
    double* data = arr.data();

    if (!data) {
        PyErr_SetString(PyExc_RuntimeError, "SharedArray has no underlying data");
        SWIG_fail;
    }

    // Create a NumPy array from the data
    npy_intp dims[1] = { static_cast<npy_intp>(size) };
    PyObject* npArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data);
    if (!npArray) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        SWIG_fail;
    }

    // Handle ownership and manage data transfer to Python
    if (arr.isPythonOwner()) {
        if (arr.own()) {
            // C++ owns the data; transfer ownership to Python and release C++ ownership
            PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
            arr.clear(); // Clear the internal state to reflect Python ownership
        } else {
            // Python owns the data; ensure NumPy takes ownership correctly
            PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
        }
    } else {
        // C++ is the owner; transfer ownership to Python
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
        arr.clear(); // Clear the internal state to release C++ ownership
    }

    // Set the result to the NumPy array
    $result = npArray;
}


/////////////////////////////////////////////////////
// C++ SharedArray2D<T> ---> Python 2D Numpy Array //
/////////////////////////////////////////////////////

%typemap(out) SharedArrayDouble2D {
    SharedArrayDouble2D arr = $1;

    // Retrieve dimensions and data from the SharedArray2D<double>
    size_t n_rows = arr.rows();
    size_t n_cols = arr.cols();
    double* data = arr.data();

    if (!data) {
        PyErr_SetString(PyExc_RuntimeError, "SharedArray2D has no underlying data");
        SWIG_fail;
    }
    // Create a NumPy array from the data
    npy_intp dims[2] = { static_cast<npy_intp>(n_rows), static_cast<npy_intp>(n_cols) };
    PyObject* npArray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, data);
    if (!npArray) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        SWIG_fail;
    }
    // Handle ownership and manage data transfer to Python
    if (arr.isPythonOwner()) {
        if (arr.own()) {
            // C++ owns the data; transfer ownership to Python and release C++ ownership
            PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
            arr.clear(); // Clear the internal state to reflect Python ownership
        } else {
            // Python owns the data; ensure NumPy takes ownership correctly
            PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
        }
    } else {
        // C++ is the owner; transfer ownership to Python
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
        arr.clear(); // Clear the internal state to release C++ ownership
    }
    // Set the result to the NumPy array
    $result = npArray;
}


//////////////////////////////////////////////////////////////////
// C++ Vector of SharedArray<T> ---> Python List of Numpy Array //
//////////////////////////////////////////////////////////////////

%typemap(out) ListSharedArrayDouble1D {
    PyObject *list = PyList_New($1.size());
    if (!list) {
        SWIG_exception_fail(SWIG_MemoryError, "Unable to allocate memory for output list");
    }
    for (size_t i = 0; i < $1.size(); ++i) {
        SharedArrayDouble1D arr = $1[i];

        npy_intp dims[1] = { static_cast<npy_intp>(arr.size()) };
        PyObject *npArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, arr.data());

        if (!npArray) {
            Py_DECREF(list);
            SWIG_exception_fail(SWIG_RuntimeError, "Unable to create numpy array");
        }
        if (arr.isPythonOwner()) {
            if (arr.own()) {
                PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
                arr.clear();
            } else {
                PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
            }
        } else {
            PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
            arr.clear();
        }
        PyList_SET_ITEM(list, i, npArray);
    }
    $result = list;
}


////////////////////////////////////////////////////////////////////////////////////
// C++ Vector of Vector of SharedArray<T> ---> Python List of List of Numpy Array //
////////////////////////////////////////////////////////////////////////////////////

%typemap(out) ListListSharedArrayDouble1D {
    // Create an outer Python list
    PyObject *outer_list = PyList_New($1.size());
    if (!outer_list) {
        SWIG_exception_fail(SWIG_MemoryError, "Unable to allocate memory for output outer list");
    }
    for (size_t i = 0; i < $1.size(); ++i) {
        // Create an inner Python list
        PyObject *inner_list = PyList_New($1[i].size());
        if (!inner_list) {
            Py_DECREF(outer_list);
            SWIG_exception_fail(SWIG_MemoryError, "Unable to allocate memory for output inner list");
        }
        for (size_t j = 0; j < $1[i].size(); ++j) {
            SharedArrayDouble1D arr = $1[i][j];

            npy_intp dims[1] = { static_cast<npy_intp>(arr.size()) };
            PyObject *npArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, arr.data());

            if (!npArray) {
                Py_DECREF(inner_list);
                Py_DECREF(outer_list);
                SWIG_exception_fail(SWIG_RuntimeError, "Unable to create numpy array");
            }
            if (arr.isPythonOwner()) {
                if (arr.own()) {
                    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
                    arr.clear();
                } else {
                    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
                }
            } else {
                PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);
                arr.clear();
            }
            PyList_SET_ITEM(inner_list, j, npArray);
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
    }
    $result = outer_list;
}
