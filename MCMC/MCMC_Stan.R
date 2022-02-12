library(lme4)
library(rstan)
library(rstanarm)
library(ggplot2)
library(tidyverse)
library(bayesplot)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

setwd('/Users/shanakaperera/Documents/PhD/Consumption_GMM/BSIM_Complete/MCMC')

### FOr loop response - lambda mu unknown Gamma inverse for sigma y
S = 10
N = 1000
truncate<-4

source('Load_Data.R')

n_features = dim(features)[2]
n_store_features = dim(store_features)[2]

input_data <- list(S = S, N = N, n_features = n_features,n_store_features= n_store_features, features = features, store_features = store_features,
                   locations_N = locations_N,locations_S = locations_S, truncate = truncate, y=y_array)

model_code <- 'data {
  int S; 
  int N; //number of rows
  int n_features; //number of columns
  int n_store_features; 
  matrix[N,n_features] features;
  matrix[S,n_store_features] store_features; 
  matrix[N,2] locations_N; 
  matrix[S,2] locations_S; 
  real truncate;
  real y[S];        // Total consumptions
}
parameters {
  vector[n_store_features] lambda_mu;  
  vector[n_features] beta;  
  real<lower=0> gamma_beta;
  real<lower=0> gamma_y;
}
transformed parameters {
  vector[N] inputs_N1 = col(locations_N,1);
  vector[N] inputs_N2 = col(locations_N,2);
  vector[S] inputs_S1 = col(locations_S,1);
  vector[S] inputs_S2 = col(locations_S,2);
  vector[S] trunc_denom ;
  vector[N] dist_sqrd; 
  vector[N] trunc_numerat; 
  vector[N] comparison;
  matrix[N,S] pdf_all;
  vector[N] pdf_evaluation_sum;
  matrix[N,S] respons;
  vector[S] y_hat;
  vector[N] f_function = features*beta;
  vector[S] tau_S =  exp(store_features* lambda_mu); 
  for (s in 1:S){
    trunc_denom[s] = 2 * pi() * tau_S[s]  * (1- exp(-truncate^2 / (2*tau_S[s] ))) ;
    dist_sqrd = square(inputs_N1 - inputs_S1[s])+ square(inputs_N2- inputs_S2[s]) ; 
    trunc_numerat = exp(-0.5/tau_S[s] * dist_sqrd);
    for(n in 1:N){
     comparison[n] = dist_sqrd[n]<truncate^2;
    }
    trunc_numerat = comparison  .* trunc_numerat;
    pdf_all[,s] = trunc_numerat / trunc_denom[s];
  }
  for (n in 1:N){
    pdf_evaluation_sum[n] = sum(row(pdf_all,n)) + 1e-10;
  }
  for (s in 1:S){
    respons[,s] = col(pdf_all,s) ./ pdf_evaluation_sum;  
  }
  for (s in 1:S){
    y_hat[s] =  sum(f_function .* col(respons, s));
  }
}
model {
  target += gamma_lpdf(gamma_beta| 1,1);       // prior gamma beta log-density
  target += normal_lpdf(beta| 0, 1/sqrt(gamma_beta));       // prior beta log-density
  target += normal_lpdf(lambda_mu| 0, 1);       // prior lamda log-density
  target += gamma_lpdf(gamma_y| 1,1);       // prior gammay y log-density
  target += normal_lpdf(y | y_hat, 1/sqrt(gamma_y)); // log-likelihood
}'



ptm <- proc.time()
m<- stan_model(model_code = model_code ,verbose = FALSE)
fit1 <- sampling(m,  data = input_data, iter = 200,warmup = 100,thin = 5,chains = 4) # algorithm = "HMC"
proc.time() - ptm
 
print(fit1, pars=c('beta', 'lambda_mu','gamma_y','gamma_beta'), digits=3, prob=c(.025,.5,.975))

rstan::traceplot(fit1 ,pars=c('beta', 'lambda_mu','gamma_y','gamma_beta'))

###-- Plots for thesis 
rstan::traceplot(fit1 ,pars=c('beta'), inc_warmup = TRUE)

fit_summary <- summary(fit1)
all_sumry <- fit_summary$c_summary
beta_summary <- summary(fit1, pars = c("beta"), probs = c(0.1, 0.9))$c_summary


plot(fit1, plotfun = "rhat")
stan_rhat(fit1)


summary(fit1)$summary

smry <- summary(fit1)$summary
stat <- switch("rhat", rhat = smry[, "Rhat"], n_eff_ratio = smry[,"n_eff"]/SS, mcse_ratio = smry[, "se_mean"]/smry[, "sd"])
df <- data.frame(stat)
df$par <- rownames(df)

  
##


### Plots of Posteriror distributions

#---Beta1


for (k in seq(1,2)){
  
  cols <- c("MCMC"="#377EB8", "True" = "#E41A1C")
  lines <- c("MCMC" = "solid", "True"  =  "dashed" )
  
  beta = rstan::extract(fit1, pars='beta')$beta
  
  x_ <- matrix(beta[,k])
  colnames(x_) <- c("MCMC")
  x <- as.tibble(x_)
  df <- x %>%  pivot_longer(MCMC, names_to = "Variable", values_to = "Values")
  
  min_<- min(x_) #vi_beta_mu[k] - 4* sqrt(vi_beta_var[k])
  max_<- max(x_) #vi_beta_mu[k] + 4* sqrt(vi_beta_var[k])

  plt <- df %>% ggplot() + 
    geom_histogram(data = df, aes(x = Values, y = ..density..) , bins = 50,colour = "#deeefa", fill = "#377EB8", alpha=0.1)+
    geom_density(aes(x = Values, colour = "MCMC"), show.legend = FALSE) +
    geom_vline(aes(xintercept = opt_beta_mu[k,1],  colour = "True"), linetype = "dashed", show.legend = FALSE) +
    scale_color_manual(name = "",values = cols) +  
    theme_bw() + 
    theme(legend.key = element_rect(fill = NA),
          legend.background=element_rect(colour= NA),
          legend.position = c(.8,.75),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),  text = element_text(size=20)) +xlab('')+ylab('')
  
  png(paste0("beta",k,".png"), width = 400, height = 350)
  print(plt)
  dev.off()
}


# -- y gamma
# Posterior
gamma_y = rstan::extract(fit1, pars='gamma_y')$gamma_y
x <- matrix(gamma_y)
colnames(x) <- c("MCMC")
x <- as.tibble(x)
df <- x %>%  pivot_longer(MCMC, names_to = "Variable", values_to = "Values")

plt<-df %>% ggplot() + 
  geom_histogram(data = df, aes(x = Values, y = ..density..) , bins = 50,colour = "#deeefa", fill = "#377EB8", alpha=0.1 )+
  geom_density(aes(x = Values, colour = "MCMC"), show.legend = FALSE) +
  geom_vline(aes(xintercept =  1/sigma_y_actual^2,  colour = "True"), linetype = "dashed", show.legend = FALSE) +
  scale_color_manual(name = "",values = cols) +  
  theme_bw() + 
  theme(legend.key = element_rect(fill = NA),
        legend.background=element_rect(colour=NA),
        legend.position = c(.8,.75),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),  text = element_text(size=20)) +xlab('')+ylab('')

png(paste0("Gamma3.png"), width =  400, height = 350)
print(plt)
dev.off()

# - lambda

for(k in seq(1,2)){

  lambda_mu = rstan::extract(fit1, pars='lambda_mu')$lambda_mu
  
  x_ <- matrix(lambda_mu[,k])
  colnames(x_) <- c("MCMC")
  x <- as.tibble(x_)
  df <- x %>%  pivot_longer(MCMC, names_to = "Variable", values_to = "Values")
  
  min_<- min(x_)  #vi_lambda_mu[k] - 4* sqrt(vi_lambda_var[k])
  max_<- max(x_) #vi_lambda_mu[k] + 4* sqrt(vi_lambda_var[k])
  
  plt <- df %>% ggplot() + 
    geom_histogram(data = df, aes(x = Values, y = ..density..) , bins = 50,colour = "#deeefa", fill = "#377EB8", alpha=0.1 )+
    geom_density(aes(x = Values, colour = "MCMC"), show.legend = FALSE) +
    geom_vline(aes(xintercept = opt_lambda_mu[k,1],  colour = "True"), linetype = "dashed", show.legend = FALSE) +
    scale_color_manual(name = "",values = cols) +  
    theme_bw() + 
    theme(legend.key = element_rect(fill = NA),
          legend.background=element_rect(colour=NA),
          legend.position = c(.8,.75),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),  text = element_text(size=20)) +xlab('')+ylab('')
        

  png(paste0("lambda",k,".png"), width =  400, height = 350)
  print(plt)
  dev.off()
  
}


# - Beeta_Gamma 


# Posterior
gamma_beta = rstan::extract(fit1, pars='gamma_beta')$gamma_beta
x <- matrix(gamma_beta)
colnames(x) <- c("MCMC")
x <- as.tibble(x)
df <- x %>%  pivot_longer(MCMC, names_to = "Variable", values_to = "Values")

df %>% ggplot() + 
  geom_density(aes(x = Values, colour = "MCMC"), show.legend = FALSE) +
  scale_color_manual(name = "",values = cols) +  
  theme_bw() + 
  theme(legend.key = element_rect(fill = NA),
        legend.background=element_rect(colour="grey"),
        legend.position = c(.9,.75),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),  text = element_text(size=20))

