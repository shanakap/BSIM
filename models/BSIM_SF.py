# BSIM_SF - Bayesian Spatial Interaction Model : Store variace as a function of the Store Features
#var_s = lambda * store_features

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from sklearn import preprocessing
import models.util as util

class BSIM(tf.Module):
    def __init__(self):
        super(BSIM, self).__init__(name='BSIM')
    """
    The class representing the MTSM model with optimised mixing weights.
    Parameters
    ----------
    n_stores:
        Number of stores - S
    n_features:
        Numper of features - P
    n_s_features:
        number of store features - m
    n_individuals:
        Number of individuals - N
    truncate:
        Radius of the truncated Gaussian
    """

    def __init__(self,
        n_individuals,
        n_stores,
        n_features,
        n_s_features,
        truncate,
        prior_beta_mu = tf.constant(0.,dtype=np.float32 ) ,
        prior_beta_gamma_a = tf.constant(1.,dtype=np.float32 ),
        prior_beta_gamma_b =tf.constant(1.,dtype=np.float32 ) ,
        prior_y_gamma_a = tf.constant(1.,dtype=np.float32 ) ,
        prior_y_gamma_b =tf.constant(1.,dtype=np.float32 ),
        prior_lambda_mu = tf.constant(0.,dtype=np.float32 ),
        prior_lambda_var = tf.constant(1.,dtype=np.float32)):

        # Initialize all model dimension constants.
        self.S = n_stores
        self.n_features = n_features
        self.n_s_features = n_s_features
        self.N = n_individuals
        self.truncate = tf.constant(truncate,dtype=np.float32 )
        self.eps = 1e-16 #add to make precision issues

        # Define the number of MonteCarlo samples used for the computation of the ell term
        self.n_samples = 10

        # Global step to use for the decay computation. Must not be negative.
        self.global_step = tf.Variable( 1 , trainable=False)


        ## Define all parameters of the prior distributions
        # Mean of Beta Prior distribution
        self.prior_beta_mu = tf.Variable(prior_beta_mu * tf.ones([self.n_features,1],dtype=np.float32),dtype=np.float32)

        # Prior distribution for precison of Beta
        self.prior_beta_gamma_a = prior_beta_gamma_a
        self.prior_beta_gamma_b = prior_beta_gamma_b

        # Prior distribution for precison of the model (1/Error term)
        self.prior_y_gamma_a = prior_y_gamma_a
        self.prior_y_gamma_b = prior_y_gamma_b

        #Prior distribution for Lambda
        self.prior_lambda_mu =  tf.Variable(prior_lambda_mu *tf.ones([self.n_s_features,1],dtype=np.float32),dtype=np.float32)
        self.prior_lambda_var = np.float32(prior_lambda_var * np.eye(self.n_s_features))

        # Define all variational parameters directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions (eg variance needs to be positive).

        # Approximate posterior distribution for hyper prior on beta distribution
        self.raw_beta_gamma_a = tf.Variable(tf.math.log(tf.constant(1.,dtype=np.float32 )),dtype=np.float32)
        self.raw_beta_gamma_b = tf.Variable(tf.math.log(tf.constant(1.,dtype=np.float32 )),dtype=np.float32)

        # Approximate posterior distribution for precision
        self.raw_y_gamma_a =  tf.Variable(tf.math.log(tf.constant(1.,dtype=np.float32 )),dtype=np.float32)
        self.raw_y_gamma_b =  tf.Variable(tf.math.log(tf.constant(1.,dtype=np.float32 )),dtype=np.float32)

        # Mean of approximate posterior distribution for beta
        self.beta_mu = tf.Variable(0. * tf.ones([self.n_features,1],dtype=np.float32),dtype=np.float32)

        # Covariance of approximate posterior distribution for beta. We initialise the lower triangular matrix of the Cholesky decomposition.
        init_vec = np.zeros([1] +  util.tri_vec_shape(self.n_features), dtype=np.float32)
        self.raw_beta_var = tf.Variable(init_vec)

        # Mean of approximate posterior distribution for lambda
        self.lambda_mu = tf.Variable(0*tf.ones([self.n_s_features,1],dtype=np.float32),dtype=np.float32)

        # Covariance of approximate posterior distribution for lambda. We initialise the lower triangular matrix of the Cholesky decomposition.
        lambda_init_vec = np.zeros([1] +  util.tri_vec_shape(self.n_s_features), dtype=np.float32)
        self.raw_lambda_var = tf.Variable(lambda_init_vec) # k_lambda m*m

    def _build_cross_ent(self,beta_gamma_a, beta_gamma_b,beta_mu,beta_cholesky,
                lambda_mu,lambda_cholesky,train_outputs,y_gamma_a,y_gamma_b):
        # This function is building the cross-entropy

        trace = tf.math.reduce_sum(tf.math.square(tf.linalg.diag_part(beta_cholesky)))
        cross_ent2 = -0.5 * self.n_features * tf.math.log(tf.constant(2 * np.pi,dtype=np.float32 )) + 0.5*self.n_features  *(tf.math.digamma(beta_gamma_a) -\
         tf.math.log(beta_gamma_b)) - 0.5*beta_gamma_a/beta_gamma_b * (trace+tf.matmul(tf.transpose(a=beta_mu),beta_mu) - 2* tf.matmul(tf.transpose(a=self.prior_beta_mu),beta_mu) + tf.matmul(tf.transpose(a=self.prior_beta_mu),self.prior_beta_mu))

        cross_ent3 = - tfp.distributions.Gamma.cross_entropy(tfp.distributions.Gamma(beta_gamma_a, beta_gamma_b),tfp.distributions.Gamma(self.prior_beta_gamma_a,self.prior_beta_gamma_b))

        cross_ent4 = - tfp.distributions.Gamma.cross_entropy(tfp.distributions.Gamma(y_gamma_a, y_gamma_b),tfp.distributions.Gamma(self.prior_y_gamma_a,self.prior_y_gamma_b))

        cross_ent5 = -tfp.distributions.MultivariateNormalTriL.cross_entropy(tfp.distributions.MultivariateNormalTriL(tf.squeeze(lambda_mu),lambda_cholesky) , tfp.distributions.MultivariateNormalTriL(tf.squeeze(self.prior_lambda_mu), tf.linalg.cholesky(self.prior_lambda_var)))

        return  cross_ent2 + cross_ent3 + cross_ent4 + cross_ent5

    def _build_ent(self,beta_gamma_a, beta_gamma_b,beta_cholesky,lambda_mu,lambda_cholesky,y_gamma_a, y_gamma_b):

        # This function is building the entropy

        # Using cholesky decomposition to compute covariance matrix determinant
        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        log_det_beta_var = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(beta_cholesky)))
        ent1 = -0.5 * log_det_beta_var - 0.5 * self.n_features*(tf.math.log(tf.constant(2 * np.pi,dtype=np.float32)) +1)

        ent2 =  - tfp.distributions.Gamma.entropy(tfp.distributions.Gamma(beta_gamma_a, beta_gamma_b))

        ent3 = - tfp.distributions.Gamma.entropy(tfp.distributions.Gamma(y_gamma_a, y_gamma_b))

        ent4 = -tfp.distributions.MultivariateNormalTriL.entropy( tfp.distributions.MultivariateNormalTriL(tf.squeeze(lambda_mu),lambda_cholesky))

        return  ent1+ ent3 + ent2 + ent4

    def _build_ell(self,n_samples,lambda_samples,beta_samples,beta_mu,
        train_inputs,train_outputs,train_inputs_sf,y_gamma_a,y_gamma_b):

        # This function is building the expected log likelihood

        truncate = self.truncate
        dist_sqrd = self.dist_sqrd
        comparison = self.comparison

        var_s_samples_clip = tf.clip_by_value(tf.matmul(train_inputs_sf, lambda_samples),-88,88)
        var_s_samples = tf.transpose(tf.exp(var_s_samples_clip))+ self.eps

        beta_x = tf.matmul(beta_samples,train_inputs)

        mc_lambda = 0
        for i in range(n_samples):
            var_s = var_s_samples[i]
            b1 = -(tf.square(truncate)/ (2.*var_s) + self.eps)
            b = tf.where(tf.abs(b1) < self.eps, self.eps * tf.ones_like(b1), b1)

            c = (1.- tf.exp(b))
            c  = tf.where(tf.abs(c) < self.eps,  tf.zeros_like(c), c)

            d = var_s * c
            trunc_denom = 2. * np.pi * d + self.eps

            trunc_numerat = tf.exp(-0.5/tf.transpose(var_s) * dist_sqrd)
            trunc_numerat = tf.multiply(trunc_numerat,comparison)

            pdf_rshp = (trunc_numerat/tf.transpose(trunc_denom))
            pdf_evaluation_sum = tf.reshape(tf.math.reduce_sum(pdf_rshp,axis = 1),[self.N,1]) + self.eps

            respons =  pdf_rshp/pdf_evaluation_sum
            beta_xs = tf.matmul(beta_x,respons)

            del respons, trunc_numerat, pdf_rshp

            y_beta_xs = tf.square(train_outputs-tf.transpose(beta_xs))
            mc_beta = tf.reduce_sum(tf.reduce_mean(y_beta_xs, axis = 1))

            mc_lambda += mc_beta

        mc_value = mc_lambda/n_samples

        ell = -0.5*self.S*tf.math.log(tf.constant(2 * np.pi,dtype=np.float32 )) + 0.5*self.S*(tf.math.digamma(y_gamma_a) - tf.math.log(y_gamma_b))- 0.5*(y_gamma_a/y_gamma_b) * mc_value

        return ell

    @tf.function
    def nelbo(self):
        raw_beta_gamma_a,raw_beta_gamma_b,raw_y_gamma_a,raw_y_gamma_b,beta_mu,raw_beta_var, lambda_mu,raw_lambda_var= self.raw_beta_gamma_a,self.raw_beta_gamma_b,self.raw_y_gamma_a,self.raw_y_gamma_b,\
                            self.beta_mu,self.raw_beta_var,self.lambda_mu,self.raw_lambda_var

        train_inputs,train_outputs,train_inputs_sf = self.train_inputs,self.train_outputs,self.train_inputs_sf

        # First transform all raw variables into their internal form. The optimisation is realized on the unconstrained variables.
        # Variables are then brought back to the acceptable regions. (eg positive values for variances)
        # The values that are restricted to be positive are :
        # -- the alpha and beta values of the Gamma distributions

        beta_gamma_a, beta_gamma_b = tf.exp(tf.clip_by_value(raw_beta_gamma_a,-88,88)),tf.exp(tf.clip_by_value(raw_beta_gamma_b,-88,88))
        y_gamma_a, y_gamma_b = tf.exp(tf.clip_by_value(raw_y_gamma_a,-88,88)),tf.exp(tf.clip_by_value(raw_y_gamma_b,-88,88))

        ### MC sampling

        # The cholesky has positive DIAGONAL entries thus we substitute the diagonal element of the cholesky
        # with their exponential in order to garantee the positive definitness.
        # We use mat_to_low_triangl_positive_diag(raw_covars) to go from one vector to a lower triangular matrix.
        # We only optimize over the lower triangular portion of the Cholesky.
        # NB. We note that we will always operate over the cholesky space internally!!!

        lambda_cholesky = util.mat_to_low_triangl_positive_diag(raw_lambda_var,self.n_s_features)
        dist_lambda = tfp.distributions.MultivariateNormalTriL(tf.squeeze(lambda_mu),lambda_cholesky)

        beta_cholesky = util.mat_to_low_triangl_positive_diag(raw_beta_var,self.n_features)
        dist_beta = tfp.distributions.MultivariateNormalTriL(tf.squeeze(beta_mu),beta_cholesky)

        tf.random.set_seed(1234)
        n_samples = self.n_samples
        lambda_samples = tf.transpose(dist_lambda.sample(sample_shape=([n_samples]),seed=123))
        beta_samples = tf.squeeze(dist_beta.sample(sample_shape=([n_samples]),seed=123))


        #### Build objective function by computing the terms composing the nelbo
        ell = self._build_ell(n_samples,lambda_samples,beta_samples,beta_mu,train_inputs,train_outputs,train_inputs_sf, y_gamma_a,y_gamma_b)

        cross_ent = self._build_cross_ent(beta_gamma_a, beta_gamma_b, beta_mu, beta_cholesky,lambda_mu,lambda_cholesky,train_outputs, y_gamma_a,y_gamma_b)
        entropy = self._build_ent(beta_gamma_a, beta_gamma_b,beta_cholesky,lambda_mu,lambda_cholesky ,y_gamma_a, y_gamma_b)

        elbo = ell + tf.squeeze(cross_ent) - entropy

        return - elbo


    def fit(self,locations_N,locations_S,features,store_features, Y_s,
        epochs=100, display_step_nelbo = 1, learning_rate = 0.05 ,decay_steps = 10,fixed_sigma_its = 0):
        """
        Fit the BSIM model to the given data.
        This function is returning the nelbo values over iterations.
        ----------
        data : Locations and Features for individuas and stores, store revenue
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        epochs : int
            The number of epochs to optimize the model for. These give the number of complete pass through the data.
        display_step_nelbo:
            The frequency at which current values are printed out and stored.
        learning_rate:	A Tensor or a floating point value.
            The initial learning rate.
        decay_steps : int
            Number of steps to decayed learning rate as a staircase function


        Returns
        ----------
        nelbo_vector: np.array
            Values of the objective function over epochs
        _vector: np.array
            Approximate posterior mean and posterior vars at every display_step_nelbo iter.
        opt_ : float
            Optimum Approximate posterior mean and posterior vars
        """

        self.train_inputs = features
        self.train_inputs_sf = store_features
        self.train_outputs = Y_s
        truncate = self.truncate

        inputs_S_expnd =  locations_S
        inputs_N_expnd =  tf.transpose(tf.reshape(tf.tile(locations_N,[self.S,1]),[self.S,self.N,2]))
        inputs_N1, inputs_N2 = inputs_N_expnd[0,:,:], inputs_N_expnd[1,:,:]
        self.dist_sqrd = tf.square(inputs_N1 - tf.transpose(inputs_S_expnd[:,0]))+ tf.square(inputs_N2 - tf.transpose(inputs_S_expnd[:,1]))
        self.comparison =  tf.cast(tf.less_equal( self.dist_sqrd, tf.square(tf.constant(truncate))), tf.float32)

        # Define the tf optimizer
        starter_learning_rate = learning_rate
        learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, self.global_step,decay_steps, 0.96, staircase=True)
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)

        ###Checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), model = self,optimizer=optimizer)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts_sf', max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # Initialise counters of epochs
        old_epoch = 0
        initial_epoch = 1

        # Define tensor to store the values of different objects over epochs
        nelbo_vector = []
        time_tensor,lr_vector = [],[]
        beta_mu_vec,beta_var_vec,lambda_mu_vec,lambda_var_vec,sigma_y_vec,y_gamma_vec, lambda_s_mu_vec,lambda_s_var_vec, beta_gamma_vec =[],[],[],[],[],[],[],[], []
        epochs_completed = 0

        # Start training phase over epochs
        train_start_time = time.time()

        while epochs_completed < epochs:
            start = time.time()

            ## Optimise parameters
            optimizer.minimize(self.nelbo, global_step=self.global_step, var_list = [self.beta_mu, self.raw_beta_var,self.raw_beta_gamma_a,self.raw_beta_gamma_b,self.lambda_mu,self.raw_lambda_var,self.raw_y_gamma_a,self.raw_y_gamma_b])


            # Time training step
            end = time.time()
            time_elapsed = end - start

            # Once the optimisation is finished, convert objects to be saved in arrays

            ckpt.step.assign_add(1)
            if (int(ckpt.step)-1) % display_step_nelbo == 0 or (int(ckpt.step)-1) ==1:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step)-1, save_path))

                nelbo_value,beta_mu,raw_beta_var,lambda_mu,raw_lambda_var,raw_beta_gamma_a,raw_beta_gamma_b,raw_y_gamma_a,raw_y_gamma_b =  self.nelbo(),self.beta_mu,self.raw_beta_var,self.lambda_mu,self.raw_lambda_var,self.raw_beta_gamma_a,self.raw_beta_gamma_b,self.raw_y_gamma_a,self.raw_y_gamma_b

                beta_cholesky = util.mat_to_low_triangl_positive_diag(raw_beta_var,self.n_features)
                beta_var = tf.matmul(beta_cholesky,tf.transpose(beta_cholesky))

                lambda_cholesky = util.mat_to_low_triangl_positive_diag(raw_lambda_var,self.n_s_features)
                lambda_var = tf.matmul(lambda_cholesky,tf.transpose(lambda_cholesky))

                y_gamma_a, y_gamma_b = tf.exp(raw_y_gamma_a), tf.exp(raw_y_gamma_b)            #
                beta_gamma_a, beta_gamma_b = tf.exp(raw_beta_gamma_a), tf.exp(raw_beta_gamma_b)

                print(' ' + 'i=' + str(epochs_completed+1))
                print(' ' + 'nelbo_value =' + str(nelbo_value.numpy()))
                print(' ' + 'beta_mu =' + str(beta_mu.numpy()))
                print(' ' + 'learning_rate =' + str(learning_rate().numpy()))

                # Append values to save them

                beta_mu_vec.append(beta_mu.numpy())
                beta_var_vec.append(beta_var.numpy())
                lambda_mu_vec.append(lambda_mu.numpy())
                lambda_var_vec.append(lambda_var.numpy())

                sigma_y = 1/np.sqrt(y_gamma_a/y_gamma_b)
                y_gamma_vec.append(np.array([y_gamma_a,y_gamma_b]))
                sigma_y_vec.append(sigma_y)
                beta_gamma_vec.append(np.array([beta_gamma_a,beta_gamma_b]))
                nelbo_vector.append(nelbo_value.numpy())
            epochs_completed+=1

        opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,\
                    opt_beta_gamma_a,opt_beta_gamma_b,opt_y_gamma_a,opt_y_gamma_b  = beta_mu.numpy(),beta_var.numpy(), lambda_mu.numpy(),lambda_var.numpy(),\
                                    beta_gamma_a.numpy(),beta_gamma_b.numpy(),y_gamma_a.numpy(),y_gamma_b.numpy()
        opt_lambda_s_mu, opt_lambda_s_var = 0,0
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        print('Training time : ' + str(train_time))


        return (nelbo_vector, time_tensor,beta_mu_vec,beta_var_vec,lambda_mu_vec,lambda_var_vec, lambda_s_mu_vec,lambda_s_var_vec,sigma_y_vec,y_gamma_vec,opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,
                    opt_lambda_s_mu,opt_lambda_s_var, opt_beta_gamma_a,opt_beta_gamma_b,opt_y_gamma_a,opt_y_gamma_b,beta_gamma_vec)

    def predict(self, inputs_N, inputs_S,features,store_features,opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_lambda_s_mu,opt_lambda_s_var,opt_y_gamma_a,opt_y_gamma_b):

        """
        After training, predict outputs given testing inputs.
        Parameters
        ----------
        test_inputs : ndarray
            Locations and features of stores which we wish to make predictions and  individuals locations and features.
        opt_ : float
            Optimal values of the variational parameters

        Returns
        -------
        y_est: ndarray
            Predicted mean for the test inputs
        n_rev_array: ndarray
            Predicted revenue at the individual level
        """
        S = inputs_S.shape[0]
        N = inputs_N.shape[0]

        n_rev_array = []
        y_array = np.zeros(S)

        batch_inputs_N = inputs_N
        batch_inputs_S = inputs_S
        batch_store_features =  store_features
        inputs_N_expnd =  tf.transpose(a=tf.reshape(tf.tile(batch_inputs_N,[S,1]),[S,N,2]))

        # Store variance
        tau_S =  np.exp(np.matmul(batch_store_features, opt_lambda_mu),dtype =  np.float32)

        # Truncated Gaussian
        trunc_denom = 2 * np.pi * tau_S * (1- tf.exp(-(self.truncate**2 / (2*tau_S ) + self.eps))) + self.eps

        inputs_N1, inputs_N2 = inputs_N_expnd[0,:,:], inputs_N_expnd[1,:,:]
        dist_sqrd = (inputs_N1 - tf.transpose(a=batch_inputs_S[:,0])) **2 + (inputs_N2- tf.transpose(a=batch_inputs_S[:,1]))**2

        trunc_numerat = tf.Variable(tf.exp(-0.5/tf.transpose(a=tau_S) * dist_sqrd))

        comparison = tf.greater( dist_sqrd, tf.constant( self.truncate**2 ) )
        trunc_numerat = trunc_numerat.assign( tf.compat.v1.where (comparison, tf.zeros_like(trunc_numerat), trunc_numerat) )

        pdf = (trunc_numerat/tf.transpose(a=trunc_denom))

        pdf_evaluation_sum = tf.reshape( tf.math.reduce_sum(input_tensor=pdf,axis = 1),[N,1]) + 1e-10

        # Probability of each individual visiting each store
        respons = pdf/pdf_evaluation_sum
        respons = respons.numpy()


        # Calculate total revenue at each store by the individuals
        for store in range(S):
            respons_store = respons[:,store][:,np.newaxis]
            f_function = np.matmul(opt_beta_mu.T, features)
            consumption_n = respons_store[:,0] * f_function[0,:]

            y_s = np.sum(consumption_n)
            y_array[store] = y_s
            n_rev_array.append(consumption_n)

        y_est = y_array.reshape(-1,1)
        return y_est, n_rev_array
