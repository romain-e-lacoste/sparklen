// Author : Romain E. Lacoste
// License : BSD-3-Clause

%module hawkes_model

%{
#define SWIG_FILE_WITH_INIT
#include "sparklen/hawkes/model/model_hawkes_exp_least_squares_single.h"
#include "sparklen/hawkes/model/model_hawkes_exp_least_squares.h"
#include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood_single.h"
#include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood.h"
%}

%include sparklen/array/array_module.i

%include "sparklen/hawkes/model/model_hawkes_exp_least_squares_single.h"
%include "sparklen/hawkes/model/model_hawkes_exp_least_squares.h"
%include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood_single.h"
%include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood.h"
