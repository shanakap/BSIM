import tensorflow as tf
import numpy as np

import tensorflow_probability as tfp
import time
from sklearn import preprocessing
# from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score
import gc
#from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
import math

import scipy

def tri_vec_shape(N):
    return [int(N * (N + 1) / 2)]


def tf_vec_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.
    Native TensorFlow version of Custom Op by Mark van der Wilk.
    def int_shape(x):
        return list(map(int, x.get_shape()))
    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)


def forward_tensor(x, N):
    """
    Transforms from the packed to unpacked representations (tf.tensors)

    :param x: packed tensor. Must have shape `self.num_matrices x triangular_number
    :return: Reconstructed tensor y of shape self.num_matrices x N x N
    """
    fwd = tf.squeeze(tf_vec_to_tri(x, N))
    return fwd


def mat_to_low_triangl_positive_diag(raw_covars,N):
    mat = forward_tensor(raw_covars, N)
    diag_mat = tf.linalg.diag(tf.linalg.diag_part(mat))
    exp_diag_mat = tf.linalg.diag(tf.exp(tf.linalg.diag_part(mat)))
    low_triangl_positive_diag = mat - diag_mat + exp_diag_mat
    return low_triangl_positive_diag

def post_process_estimates(opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_y_gamma_a, opt_y_gamma_b, beta_act = np.array([]), lambda_act= np.array([]),sigma_y_act = np.nan):
    stats_ = []
    i = 0
    for i in range(len(opt_beta_mu)):
        mu = opt_beta_mu[i]
        std  = np.sqrt(opt_beta_var[i,i])
        ci_l,ci_u = mu - 1.96*std, mu + 1.96*std
        if beta_act.shape[0]==0:
            stats_.append({'Parameter': 'Beta' + str(i), 'mu':mu[0], 'std':std, 'CI_l': ci_l[0], 'CI_u':ci_u[0] })
        else:
            true_val = beta_act[i,0]
            stats_.append({'Parameter': 'Beta' + str(i),'True' :true_val , 'mu':mu[0], 'std':std, 'CI_l': ci_l[0], 'CI_u':ci_u[0] })

    for i in range(len(opt_lambda_mu)):
        mu = opt_lambda_mu[i]
        std  = opt_lambda_var[i,i]
        ci_l,ci_u = mu - 1.96* std, mu + 1.96 * std
        if lambda_act.shape[0]==0:
            stats_.append({'Parameter': 'Lambda' + str(i),'mu':mu[0], 'std':std, 'CI_l': ci_l[0], 'CI_u':ci_u[0] })
        else:
            stats_.append({'Parameter': 'Lambda' + str(i),'True':lambda_act[i,0] ,'mu':mu[0], 'std':std, 'CI_l': ci_l[0], 'CI_u':ci_u[0] })

    gam_dist = stats.gamma(a=opt_y_gamma_a, loc=0., scale=1/opt_y_gamma_b)
    mu = gam_dist.mean()
    std = np.sqrt(gam_dist.var())
    ci_l,ci_u  = gam_dist.ppf(0.025), gam_dist.ppf(0.975)
    if np.isnan(sigma_y_act):
        stats_.append({'Parameter': 'Gamma_y','mu':mu, 'std':std, 'CI_l': ci_l, 'CI_u':ci_u})
    else:
        stats_.append({'Parameter': 'Gamma_y', 'True' :1/sigma_y_act**2, 'mu':mu, 'std':std, 'CI_l': ci_l, 'CI_u':ci_u})
    return stats_


def post_process_results(y_est,Y):
    stats_ = []
    rmse = np.round(np.linalg.norm(y_est - Y) / np.sqrt(len(Y)),2)
    nrmse = np.round(rmse/np.mean(Y),2)
    r_squred = round(r2_score(Y,y_est),2)
    stats_.append({'RMSE': rmse, 'NRMSE' : nrmse, 'R_squred':r_squred})
    return stats_


def plot_simga(sigma_y_vec,sigma_y_actual= np.nan):
    sigma_y_vec = np.asarray(sigma_y_vec)
    x = sigma_y_vec.shape[0]

    fig, ax = plt.subplots()

    ax.plot(sigma_y_vec)
    if np.isnan(sigma_y_actual):
        ax.hlines(sigma_y_actual,0,x-1,linestyles ='dashed',color = 'r')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('sigma_y')
    return fig

def plot_beta(beta_mu_vec,beta_var_vec,n_features,beta_act= np.array([])):
    x = np.asarray(beta_mu_vec).shape[0]
    fig, ax = plt.subplots()
    for i in range(n_features):
        b1 = np.asarray(beta_mu_vec)[:,i,0]
        var1 = np.asarray(beta_var_vec)[:,i,i]
        ax.plot(range(x),b1)
        ax.fill_between(range(x), b1 - 2*np.sqrt(var1), b1 + 2*np.sqrt(var1), color='blue', alpha=0.2)

    if beta_act.shape[0]>0:
        ax.hlines(beta_act,0,x-1,linestyles ='dashed',color = 'r')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('E(Beta)')
    return fig

def plot_sumary(Y,y_est):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Y,y_est)
    y_vals = np.concatenate((Y,y_est),axis=0)

    ident = [np.min(y_vals), np.max(y_vals)]
    ax.plot(ident,ident,linestyle='dashed',c='r')
    ax.set_ylabel('Y Estimated')
    ax.set_xlabel('Y True')
    return fig

def plot_lambda_synth(lambda_mu_vec,lambda_var_vec,store_features, opt_lambda_mu, var_s_actual,s_features,act_lambda):
    x = np.asarray(lambda_mu_vec).shape[0]

    figMain = plt.figure(figsize=(8, 4))
    figMain.suptitle('Lambda')
    ax = figMain.subplots(1, 2)

    col = ax[0]
    for i in range(s_features):
        b1 = np.asarray(lambda_mu_vec)[:,i,0]
        var1 = np.asarray(lambda_var_vec)[:,i,i]
        col.plot(range(x),b1)
        col.fill_between(range(x), b1 - 2*np.sqrt(var1), b1 + 2*np.sqrt(var1), color='blue', alpha=0.2)

    col.hlines(act_lambda,0,x-1,linestyles ='dashed',color = 'r')
    col.fill_between(range(x), b1 - 2*np.sqrt(var1), b1 + 2*np.sqrt(var1), color='blue', alpha=0.2)
    col.set_xlabel('Epochs')
    col.set_ylabel('E(lambda)')

    pred_sigma_s = np.exp(np.matmul(store_features, opt_lambda_mu))# (opt_lambda_mu*store_features)**2

    sigma_S_all = np.concatenate((pred_sigma_s,var_s_actual),axis = 0)

    col = ax[1]
    col.scatter(var_s_actual,pred_sigma_s)
    ident = [np.min(sigma_S_all)*0.8, np.max(sigma_S_all)*1.2]
    col.plot(ident,ident,linestyle='dashed',c='r')
    col.set_ylabel('Predicted var_s')
    col.set_xlabel('Actual var_s')

    figMain.tight_layout()
    figMain.subplots_adjust(top=0.88)
    return figMain

def plot_nelbo(nelbo_vector):
    x1 = np.asarray(nelbo_vector).shape[0]
    figMain = plt.figure(figsize=(8, 4))
    figMain.suptitle('NELBO')
    ax = figMain.subplots(1, 2)

    col = ax[0]
    col.plot(np.asarray(nelbo_vector))
    col.set_xlabel('Epochs (All)')
    col.set_ylabel('NELBO')

    col = ax[1]
    a = np.asarray(nelbo_vector)[np.int_(x1*0.5):]
    col.plot(a)
    col.set_xlabel('Epochs (last half)')
    col.set_ylabel('NELBO')
    figMain.tight_layout()
    figMain.subplots_adjust(top=0.88)
    return figMain



def plot_lambda_real(lambda_mu_vec,lambda_var_vec,opt_lambda_mu,lambda_s_mu_vec,opt_lambda_s_mu,store_features,s_features):
    if len(lambda_mu_vec)>0:
        x = range(np.asarray(lambda_mu_vec).shape[0])
    else:
        x = range(np.asarray(lambda_s_mu_vec).shape[0])

    figMain = plt.figure(figsize=(8, 4))
    figMain.suptitle('Variance of the Gaussian on Store')
    ax = figMain.subplots(1, 2)

    col = ax[0]
    if len(lambda_mu_vec)>0:
        for i in range(s_features):
            b1 = np.asarray(lambda_mu_vec)[:,i,0]
            var1 = np.asarray(lambda_var_vec)[:,i,i]
            col.plot(x,b1)#,c =colorz[i])#,label=store_features_names[i])
            col.fill_between(x, b1 - 2*np.sqrt(var1), b1 + 2*np.sqrt(var1), color='blue', alpha=0.2)

        col.set_xlabel('Epochs')
        col.set_ylabel('E(lambda)')
    else:
        col.plot(np.asarray(lambda_s_mu_vec)[:,:,0])
        col.set_xlabel('Epochs')
        col.set_ylabel('E(lambda_s)')

    col = ax[1]
    if len(lambda_mu_vec)>0:
        pred_sigma_s = np.exp(np.matmul(opt_lambda_mu.T,store_features.T)+ opt_lambda_s_mu.T)
        plt.hist(pred_sigma_s.T)
    else:
        pred_sigma_s = np.exp(opt_lambda_s_mu)
        col.hist(pred_sigma_s)
    col.set_xlabel('Var_s')
    col.set_ylabel('Frequency')

    figMain.tight_layout()
    figMain.subplots_adjust(top=0.88)
    return figMain
