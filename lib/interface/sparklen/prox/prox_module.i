// Author : Romain E. Lacoste
// License : BSD-3-Clause

%module prox

%{
#define SWIG_FILE_WITH_INIT
#include "sparklen/prox/prox_zero.h"
#include "sparklen/prox/prox_l1.h"
#include "sparklen/prox/prox_l2.h"
#include "sparklen/prox/prox_elastic_net.h"
%}

%include sparklen/array/array_module.i

%include "sparklen/prox/prox_zero.h"
%include "sparklen/prox/prox_l1.h"
%include "sparklen/prox/prox_l2.h"
%include "sparklen/prox/prox_elastic_net.h"
