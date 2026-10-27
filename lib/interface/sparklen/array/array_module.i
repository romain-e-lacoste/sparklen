// Author : Romain E. Lacoste
// License : BSD-3-Clause

%module array

%{
#define SWIG_FILE_WITH_INIT
#include "sparklen/array/array.h"
#include "sparklen/array/array2D.h"
#include "sparklen/array/sharedarray.h"
#include "sparklen/array/sharedarray2D.h"
#include <numpy/arrayobject.h>
#include <memory>
%}

// Include the NumPy interface
%include numpy.i
%include <std_vector.i>

%init %{
import_array();
%}

// Include the typemaps interface files
%include array_typemaps_in.i
%include array_typemaps_out.i
%include sharedarray_typemaps_in.i
%include sharedarray_typemaps_out.i

// Include the header file with your functions and Array class
%include "sparklen/array/array.h"
%include "sparklen/array/array2D.h"
%include "sparklen/array/sharedarray.h"
%include "sparklen/array/sharedarray2D.h"
