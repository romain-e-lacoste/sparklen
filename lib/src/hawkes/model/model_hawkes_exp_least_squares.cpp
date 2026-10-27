// Author : Romain E. Lacoste
// License : BSD-3-Clause

#include "sparklen/hawkes/model/model_hawkes_exp_least_squares.h"
#include <iostream>


// Constructor
ModelHawkesExpLeastSquares::ModelHawkesExpLeastSquares(){
	n_repetitions = 0;
	n_components = 0;
	multivariate_model = std::vector<ModelHawkesExpLeastSquaresSingle>(n_repetitions);
	for (size_t rep=0; rep<n_repetitions; ++rep){
		multivariate_model[rep] = ModelHawkesExpLeastSquaresSingle(n_components);
	}
	multivariate_model_computed = false;

}
ModelHawkesExpLeastSquares::ModelHawkesExpLeastSquares(size_t n_rep, size_t n_comp){

	n_repetitions = n_rep;
	n_components = n_comp;
	multivariate_model = std::vector<ModelHawkesExpLeastSquaresSingle>(n_repetitions);
	for (size_t rep=0; rep<n_repetitions; ++rep){
		multivariate_model[rep] = ModelHawkesExpLeastSquaresSingle(n_components);
	}
	multivariate_model_computed = false;
}

double ModelHawkesExpLeastSquares::compute_averaged_loss(const ListListSharedArrayDouble1D &list_jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta){

	double averaged_loss{0.};

	for (size_t rep=0; rep<n_repetitions; ++rep){
		averaged_loss += multivariate_model[rep].compute_loss(list_jump_times[rep], end_time, decay, theta);
	}
	averaged_loss /= n_repetitions;
	return averaged_loss;
}

SharedArrayDouble2D ModelHawkesExpLeastSquares::compute_averaged_grad(const ListListSharedArrayDouble1D &list_jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta){

	SharedArrayDouble2D averaged_grad(n_components, n_components+1);

	for (size_t rep=0; rep<n_repetitions; ++rep){
		averaged_grad.add(multivariate_model[rep].compute_grad(list_jump_times[rep], end_time, decay, theta));
	}
	averaged_grad /= n_repetitions;

	return averaged_grad;
}

SharedArrayDouble2D ModelHawkesExpLeastSquares::compute_averaged_hessian(const ListListSharedArrayDouble1D &list_jump_times, const double end_time, const double decay){

	SharedArrayDouble2D averaged_hessian(n_components+1, n_components+1);

	for (size_t rep=0; rep<n_repetitions; ++rep){
		averaged_hessian.add(multivariate_model[rep].compute_hessian(list_jump_times[rep], end_time, decay));
	}
	averaged_hessian /= n_repetitions;

	return averaged_hessian;
}
