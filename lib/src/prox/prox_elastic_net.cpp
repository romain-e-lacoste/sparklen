// Author : Romain E. Lacoste
// License : BSD-3-Clause

#include "sparklen/prox/prox_elastic_net.h"

// Constructor
ProxElasticNet::ProxElasticNet(double l1_r, double pen_const, size_t st, size_t ed, bool pos){
	l1_ratio = l1_r;
	penalization_const = pen_const;
	start = st;
	end = ed;
	positive = pos;
}

double ProxElasticNet::apply_single(const double coeff, const double step_size) {
	double threshold = step_size * l1_ratio * penalization_const;
	if (coeff > 0) {
		if (coeff > threshold) {
			return (coeff - threshold) / (1 + step_size * penalization_const * (1 - l1_ratio));
	    } else {
	    	return 0;
	    }
	} else {
		if (positive) {
			return 0;
	    } else {
	    	if (coeff < -threshold) {
	    		return (coeff + threshold) / (1 + step_size * penalization_const * (1 - l1_ratio));;
	    	} else {
	    		return 0;
	    	}
	    }
	}
}

void ProxElasticNet::apply(SharedArrayDouble2D &x, const double step_size) {
    size_t n_rows = x.rows();
    size_t n_cols = x.cols();

    if (n_cols < end) throw std::invalid_argument("The number of columns in x must exceed the value of the end attribute.");

    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = start; j < end; ++j) {
        	x(i,j) = apply_single(x(i,j), step_size);
        }
    }
}
