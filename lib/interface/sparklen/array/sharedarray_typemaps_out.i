// Author : Romain E. Lacoste
// License : BSD-3-Clause

// Arrays returned by the C++ core are handed to NumPy without copying the
// data. The NumPy array points to the C++ buffer and does not own it: its base
// is a capsule holding a copy of the C++ array's std::shared_ptr, which keeps
// the buffer alive for the lifetime of the NumPy array and its views. When the
// array and its views are no longer alive, NumPy releases the capsule: the
// capsule deletes its shared_ptr and, as the last owner, the C++ array's
// destructor frees the buffer with delete[], matching the new[] that
// allocated it.
// The data is copied into a new NumPy array instead when the array is empty,
// when C++ does not own the buffer (the capsule could not keep it alive), or
// when another C++ handle still refers to it. Either way, the C++ core keeps
// no handle on the returned data: it changes only if the array is passed back
// to the C++ core.

%{
#include <cstring>
#include <memory>
#include <new>

// Capsule destructor, called when Python releases the capsule: delete its
// shared_ptr to the C++ array
template <typename A>
static void sparklen_release_array(PyObject *capsule) {
    // Defensive: a destructor must not set a Python exception, and
    // PyCapsule_IsValid sets none (every capsule here has this name)
    if (PyCapsule_IsValid(capsule, "sparklen.array")) {
        delete static_cast<std::shared_ptr<A> *>(PyCapsule_GetPointer(capsule, "sparklen.array"));
    }
}

// Return a new NumPy array of the given shape with the data of arr (a
// SharedArray or a SharedArray2D), without copying the data when possible, or
// NULL with a Python error set.
template <typename S, size_t ND>
static PyObject *sparklen_to_numpy(const S &arr, const size_t (&shape)[ND]) {
    auto ptr = arr.getSharedPtr();
    // The shape must describe the buffer exactly: each dimension and their
    // product must fit in npy_intp, and the product must equal the number of
    // elements the C++ array holds. The size in bytes then fits too, since the
    // C++ array was allocated with that many elements.
    const int nd = static_cast<int>(ND);
    npy_intp dims[ND];
    npy_intp size = 1;
    for (size_t k = 0; k < ND; ++k) {
        if (shape[k] > static_cast<size_t>(NPY_MAX_INTP) ||
            (shape[k] != 0 && size > NPY_MAX_INTP / static_cast<npy_intp>(shape[k]))) {
            PyErr_SetString(PyExc_RuntimeError, "SharedArray shape is too large");
            return NULL;
        }
        dims[k] = static_cast<npy_intp>(shape[k]);
        size *= dims[k];
    }
    if (static_cast<size_t>(size) != arr.size()) {
        PyErr_SetString(PyExc_RuntimeError, "SharedArray shape does not match its data");
        return NULL;
    }
    if (size > 0 && !arr.data()) {
        PyErr_SetString(PyExc_RuntimeError, "SharedArray has no underlying data");
        return NULL;
    }
    // Hand the buffer over only if C++ owns it and no other handle refers to it:
    // use_count() counts arr and its copy ptr, so 2 means that no other C++
    // object can reach the buffer
    bool hand_over = size > 0 && arr.own() && !arr.isPythonOwner() && ptr.use_count() == 2;
    if (!hand_over) {
        // Copy into memory that NumPy allocates and frees
        PyObject *npArray = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
        if (npArray && size > 0) {
            std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(npArray)),
                        arr.data(), arr.size() * sizeof(double));
        }
        return npArray;
    }

    PyObject *npArray = PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, arr.data());
    if (!npArray) {
        return NULL;
    }
    // The capsule holds a copy of the shared_ptr: a reference of its own, which
    // outlives arr and ptr
    typedef typename decltype(ptr)::element_type Owned;
    // nothrow: a C++ exception must not cross into the Python interpreter
    auto *owner = new (std::nothrow) std::shared_ptr<Owned>(ptr);
    if (!owner) {
        Py_DECREF(npArray);
        return PyErr_NoMemory();
    }
    PyObject *capsule = PyCapsule_New(owner, "sparklen.array", sparklen_release_array<Owned>);
    if (!capsule) {
        delete owner;
        Py_DECREF(npArray);
        return NULL;
    }
    // PyArray_SetBaseObject steals the reference to the capsule, also on failure
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(npArray), capsule) < 0) {
        Py_DECREF(npArray);
        return NULL;
    }
    return npArray;
}
%}

////////////////////////////////////////////////
// C++ SharedArray<T> ---> Python NumPy Array //
////////////////////////////////////////////////

%typemap(out) SharedArrayDouble1D {
    const SharedArrayDouble1D &arr = $1;
    const size_t shape[1] = { arr.size() };
    $result = sparklen_to_numpy(arr, shape);
    if (!$result) {
        SWIG_fail;
    }
}


/////////////////////////////////////////////////////
// C++ SharedArray2D<T> ---> Python 2D NumPy Array //
/////////////////////////////////////////////////////

%typemap(out) SharedArrayDouble2D {
    const SharedArrayDouble2D &arr = $1;
    const size_t shape[2] = { arr.rows(), arr.cols() };
    $result = sparklen_to_numpy(arr, shape);
    if (!$result) {
        SWIG_fail;
    }
}


//////////////////////////////////////////////////////////////////
// C++ Vector of SharedArray<T> ---> Python List of NumPy Array //
//////////////////////////////////////////////////////////////////

%typemap(out) ListSharedArrayDouble1D {
    const ListSharedArrayDouble1D &arrays = $1;
    PyObject *list = PyList_New(arrays.size());
    if (!list) {
        SWIG_fail;  // PyList_New has set MemoryError
    }
    for (size_t i = 0; i < arrays.size(); ++i) {
        const size_t shape[1] = { arrays[i].size() };
        PyObject *npArray = sparklen_to_numpy(arrays[i], shape);
        if (!npArray) {
            Py_DECREF(list);
            SWIG_fail;
        }
        PyList_SET_ITEM(list, i, npArray);
    }
    $result = list;
}


////////////////////////////////////////////////////////////////////////////////////
// C++ Vector of Vector of SharedArray<T> ---> Python List of List of NumPy Array //
////////////////////////////////////////////////////////////////////////////////////

%typemap(out) ListListSharedArrayDouble1D {
    const ListListSharedArrayDouble1D &arrays = $1;
    // Create an outer Python list
    PyObject *outer_list = PyList_New(arrays.size());
    if (!outer_list) {
        SWIG_fail;  // PyList_New has set MemoryError
    }
    for (size_t i = 0; i < arrays.size(); ++i) {
        // Create an inner Python list, owned by the outer list from now on
        PyObject *inner_list = PyList_New(arrays[i].size());
        if (!inner_list) {
            Py_DECREF(outer_list);
            SWIG_fail;  // PyList_New has set MemoryError
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
        for (size_t j = 0; j < arrays[i].size(); ++j) {
            const size_t shape[1] = { arrays[i][j].size() };
            PyObject *npArray = sparklen_to_numpy(arrays[i][j], shape);
            if (!npArray) {
                Py_DECREF(outer_list);
                SWIG_fail;
            }
            PyList_SET_ITEM(inner_list, j, npArray);
        }
    }
    $result = outer_list;
}
