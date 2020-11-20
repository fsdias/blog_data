// cov_GPL2 macro from McElreath
functions{
    matrix cov_GPL2(matrix x, real sq_alpha, real sq_rho, real delta) {
        int N = dims(x)[1];
        matrix[N, N] K;
        for (i in 1:(N-1)) {
          K[i, i] = sq_alpha + delta;
          for (j in (i + 1):N) {
            K[i, j] = sq_alpha * exp(-sq_rho * square(x[i,j]) );
            K[j, i] = K[i, j];
          }
        }
        K[N, N] = sq_alpha + delta;
        return K;
    }
}
data {
  int<lower=0> N;
  int<lower=0> N_regime;
  int<lower=0> N_estate;
  int<lower=1> obs[N];
  vector[N] svap;
  int<lower=1,upper=N_estate> estate_id[N];
  int<lower=1,upper=N_regime> regime_id[N];
  matrix[N, N] dist;
  }
parameters {
vector[N_regime] regime;
vector[N_estate] estate;
real mu_regime;
real mu_estate;
real<lower=0> sigma_dist;
real<lower=0> sigma_regime;
real<lower=0> sigma_estate;
vector[N] z;

real<lower=0> etasq;
real<lower=0> rhosq;
}

transformed parameters{
vector[N] mu;
//GP stuff
vector[N] k;
matrix[N,N] L_SIGMA;
matrix[N,N] SIGMA;
SIGMA = cov_GPL2(dist, etasq, rhosq, 0.01);
L_SIGMA = cholesky_decompose(SIGMA);
k = L_SIGMA * z;
mu= regime[regime_id]+estate[estate_id]+k[obs];
}

model {
  //GP stuff

  rhosq ~ exponential( 0.5 );
  etasq ~ exponential( 2 );
  z ~ normal( 0 , 1 );

  //priors
   regime~normal(mu_regime,sigma_regime);
   estate~normal(mu_estate,sigma_estate);
   mu_regime~normal(3,2);
   mu_estate~normal(3,2);
   
   sigma_dist~exponential(1);
   sigma_regime~exponential(1);
   sigma_estate~exponential(1);
   
   
   //Likelihood
   svap ~ normal(mu , sigma_dist);
   
}
generated quantities{
real svap_pred[N] = normal_rng(mu,sigma_dist);
real mu_r4_3 = regime[4] -regime[3];
real mu_r3_2 = regime[3] -regime[2];
real mu_r3_1 = regime[3] -regime[1];
real mu_r2_1 = regime[3] -regime[1];
real res[N];

for(i in 1:N){
  res[i] = svap_pred[i]-svap[i];
  
}


}
