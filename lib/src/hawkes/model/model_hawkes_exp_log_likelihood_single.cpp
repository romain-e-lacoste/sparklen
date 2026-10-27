// Author : Romain E. Lacoste
// License : BSD-3-Clause

#include "sparklen/hawkes/model/model_hawkes_exp_log_likelihood_single.h"
#include <math.h>

// Constructor
ModelHawkesExpLogLikelihoodSingle::ModelHawkesExpLogLikelihoodSingle(){
	n_components = 0;
	weights_computed = false;
	N = ArrayInt1D(n_components);
	I = ArrayDouble1D(n_components);
	ListArrayDouble2D Psi;
}

// Overloaded constructor
ModelHawkesExpLogLikelihoodSingle::ModelHawkesExpLogLikelihoodSingle(size_t n){
	n_components = n;
	weights_computed = false;
	N = ArrayInt1D(n_components);
	I = ArrayDouble1D(n_components);
	ListArrayDouble2D Psi;
}

size_t ModelHawkesExpLogLikelihoodSingle::get_n_components(){
	return n_components;
}

void ModelHawkesExpLogLikelihoodSingle::compute_weights(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay){

	for (size_t j=0; j<n_components; ++j){
		const SharedArrayDouble1D &jump_times_j = jump_times[j];
		N[j] = jump_times_j.size();
		Psi.push_back(ArrayDouble2D(N[j], n_components));
		for (size_t j2=0; j2<n_components; ++j2){
			const SharedArrayDouble1D &jump_times_j2 = jump_times[j2];
			size_t h2 = 0;
			double G = 0;
			for (int l=0; l<jump_times_j.size(); ++l){
				if (l > 0) {
					G *= exp(-decay*(jump_times_j[l]-jump_times_j[l-1]));
				}
				while (h2 < jump_times_j2.size() && jump_times_j2[h2] <jump_times_j[l]){
					G += decay*(exp(-decay*(jump_times_j[l]-jump_times_j2[h2])));
					h2++;
				}
				Psi[j](l,j2) += G;
			}
		}
		for (size_t h=0; h<jump_times_j.size(); ++h){
			I[j] += 1-exp(-decay*(end_time-jump_times_j[h]));
		}
	}
	weights_computed = true;
}

double ModelHawkesExpLogLikelihoodSingle::compute_loss_i(const size_t i, const ListSharedArrayDouble1D &jump_times, double end_time, const double decay, const SharedArrayDouble2D &theta, const bool neg){

	double ll_value_i{0.};

	double term_integral{0.};
	double term_stieltjes{0.};

	for (int l=0; l<N[i]; ++l){
		double term_in_log = theta(i,0);
		for (size_t j=0; j<n_components; ++j){
			term_in_log += theta(i,j+1)*Psi[i](l,j);
		}
		term_stieltjes += log(term_in_log);
	}

	for (size_t j=0; j<n_components; ++j){
		term_integral += theta(i,j+1)*I[j];
	}
	ll_value_i += -theta(i,0)*end_time - term_integral + term_stieltjes;

	if (neg){
		ll_value_i *= -1;
	}

	return ll_value_i;
}


double ModelHawkesExpLogLikelihoodSingle::compute_loss(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, const bool neg){

	double loss{0.};

	if (!weights_computed){
		compute_weights(jump_times, end_time, decay);
	}

	for (size_t i=0; i<n_components; ++i){
		loss += ModelHawkesExpLogLikelihoodSingle::compute_loss_i(i, jump_times, end_time, decay, theta, neg);
	}

	return loss;
}


void ModelHawkesExpLogLikelihoodSingle::compute_grad_i(const size_t i, const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, SharedArrayDouble2D &grad, const bool neg){

	grad(i,0) = -end_time;
	for (int l=0; l<N[i]; ++l){
		double term_denom = theta(i,0);
		for (size_t j2=0; j2<n_components; ++j2){
			term_denom += theta(i,j2+1)*Psi[i](l,j2);
		}
		grad(i,0) += 1/term_denom;
	}
	if (neg){
		grad(i,0) *= -1;
	}

	for (size_t j=0; j<n_components; ++j){
		for (int l=0; l<N[i]; ++l){
			double term_denom = theta(i,0);
			for (size_t j2=0; j2<n_components; ++j2){
				term_denom += theta(i,j2+1)*Psi[i](l,j2);
			}
			grad(i,j+1) += Psi[i](l,j)/term_denom;
		}
		grad(i,j+1) -= I[j];
		if (neg){
			grad(i,j+1) *= -1;
		}
	}
}

SharedArrayDouble2D ModelHawkesExpLogLikelihoodSingle::compute_grad(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, const bool neg){

	SharedArrayDouble2D grad(n_components, n_components+1);

	if (!weights_computed){
		compute_weights(jump_times, end_time, decay);
	}

	for (size_t i=0; i<n_components; ++i){
		ModelHawkesExpLogLikelihoodSingle::compute_grad_i(i, jump_times, end_time, decay, theta, grad, neg);
	}
	return grad;
}
