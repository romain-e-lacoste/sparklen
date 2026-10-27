// Author : Romain E. Lacoste
// License : BSD-3-Clause

#include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood.h"
#include <iostream>


// Constructor
ModelHawkesExpLogLikelihood::ModelHawkesExpLogLikelihood(){
	n_repetitions = 0;
	n_components = 0;
	multivariate_model = std::vector<ModelHawkesExpLogLikelihoodSingle>(n_repetitions);
	for (size_t rep=0; rep<n_repetitions; ++rep){
		multivariate_model[rep] = ModelHawkesExpLogLikelihoodSingle(n_components);
	}
	multivariate_model_computed = false;
}
ModelHawkesExpLogLikelihood::ModelHawkesExpLogLikelihood(size_t n_rep, size_t n_comp){

	n_repetitions = n_rep;
	n_components = n_comp;
	multivariate_model = std::vector<ModelHawkesExpLogLikelihoodSingle>(n_repetitions);
	for (size_t rep=0; rep<n_repetitions; ++rep){
		multivariate_model[rep] = ModelHawkesExpLogLikelihoodSingle(n_components);
	}
	multivariate_model_computed = false;
}

double ModelHawkesExpLogLikelihood::compute_averaged_loss(const ListListSharedArrayDouble1D &list_jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, const bool neg){

	double averaged_loss{0.};

	for (size_t rep=0; rep<n_repetitions; ++rep){
		averaged_loss += multivariate_model[rep].compute_loss(list_jump_times[rep], end_time, decay, theta, neg);
	}

	return averaged_loss;
}

SharedArrayDouble2D ModelHawkesExpLogLikelihood::compute_averaged_grad(const ListListSharedArrayDouble1D &list_jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, const bool neg){

	SharedArrayDouble2D averaged_grad(n_components, n_components+1);

	for (size_t rep=0; rep<n_repetitions; ++rep){
		averaged_grad.add(multivariate_model[rep].compute_grad(list_jump_times[rep], end_time, decay, theta, neg));
	}

	return averaged_grad;
}
