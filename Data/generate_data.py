import sys

import numpy as np
import itertools
import scipy.stats as stats
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from matplotlib import pyplot as plt

def generate_synthetic_data(n_individuals , n_features ,n_stores ,s_features, sigma_y_actual,truncate, beta_act =2,lambda_act =0.5 , i=123):
    """
    n_individuals -  Number of individuals
    n_features - Number of features
    n_stores -  Number of stores
    s_features -  Number of  store features
    locations_N - individuals locations
    locations_S - Store locations
    sigma_y_actual - actual error term
    truncate - Radius of the truncated Gaussian
    """
    print('Generating synthetic data')

    #Actual Parameters
    act_lambda = lambda_act
    beta = beta_act
    sigma_y_actual = sigma_y_actual

    #### Creating locations for stores and individuals
    N, S = n_individuals, n_stores
    np.random.seed(2)
    max_cords = 10
    latitude = np.linspace(0,max_cords,np.int(np.sqrt(1000))+50).tolist()
    longitude = np.linspace(0,max_cords,np.int(np.sqrt(1000))+50).tolist()
    locations = np.random.permutation(list(itertools.product(latitude, longitude)))
    locations_N = np.array(locations[:N], dtype=np.float32)

    locations = np.array(locations[N:], dtype=np.float32) # This is to make sure location_s = location_n
    sub_loc = (locations>max_cords*0.1) & (locations<max_cords*0.9) # Removing stores in the edge
    locations_x = locations[sub_loc.prod(axis=1)==1] # This is use to generate stores inside without on edges

    np.random.seed(i)
    locations_S  = np.float32(locations_x[np.random.choice(len(locations_x), S)])


    ### Individual Features
    data_source = "../Data/"
    data_features = pd.read_csv(data_source+"individuals_sim_features.csv") #Simulated individual features for different spatial correlation structures
    locations_N_df = pd.DataFrame(locations_N)
    locations_N_df.columns = ['x','y']
    locations_N_features = pd.merge(locations_N_df,data_features, on=['x','y'])

    features = locations_N_features.iloc[:,2:2+n_features].values #.sim1.values #locations_N_features.iloc[:, 2:2+n_features]
    transform_scaler = preprocessing.MinMaxScaler() # Features are normalised. To Standardize use - StandardScaler()
    features = transform_scaler.fit_transform(features)
    features = np.float32(features.T)

    #Store Features generated from Gamma distribution
    np.random.seed(123)
    store_features = np.float32(np.random.gamma(1,1,S*s_features).reshape(-1,s_features))

    # Store specific variance as a function of store features
    var_s_actual = np.array(np.exp(np.matmul(store_features, act_lambda)),dtype=np.float32)

    #Truncated Gaussian calculations
    inputs_N_expnd =  tf.transpose(a=tf.reshape(tf.tile(locations_N,[S,1]),[S,N,2]))
    trunc_denom = 2 * np.pi * var_s_actual * (1- tf.exp(-tf.square(truncate)/ (2*var_s_actual)))
    inputs_N1, inputs_N2 = inputs_N_expnd[0,:,:], inputs_N_expnd[1,:,:]
    dist_sqrd = tf.square(inputs_N1 - tf.transpose(a=locations_S[:,0])) + tf.square(inputs_N2-   tf.transpose(a=locations_S[:,1]))
    trunc_numerat = tf.Variable(tf.exp(-0.5/tf.transpose(a=var_s_actual) * dist_sqrd))

    comparison = tf.less_equal( dist_sqrd, tf.square(tf.constant(truncate)))
    trunc_numerat = trunc_numerat*tf.cast(comparison, tf.float32)
    pdf = (trunc_numerat/tf.transpose(a=trunc_denom))
    pdf_evaluation_sum = tf.reshape( tf.math.reduce_sum(input_tensor=pdf,axis = 1),[N,1]) + 1e-10

    # Probability of each individual visiting each store
    respons = pdf/pdf_evaluation_sum

    #Store revenue calculation using the proposed model
    consumption_list,pdf_array =[],[]

    for store in range(S):
        pi_pdf = respons[:,store][:,np.newaxis]  # Probability of each individual visiting each store S

        f_function = np.matmul(beta.T, features) # Total consumption of each individual
        consumption_n = pi_pdf[:,0] * f_function[0,:] # Consumption of each individual at store S

        C_store = np.sum(consumption_n)     # Total consumption at Store S
        consumption_list.append(C_store)
        pdf_array.append(pi_pdf)

    ϵ_dist = stats.norm(loc = 0.0, scale = sigma_y_actual)
    ϵ = ϵ_dist.rvs([S])

    Y = consumption_list + ϵ # Sample Y from the outer normal distribution

    Y = Y.reshape(-1,1)
    Y = np.float32(Y)
    return(Y,locations_N,locations_S,features,var_s_actual,sigma_y_actual,store_features,act_lambda,pdf_array)


# n_stores, n_individuals, truncate,sector,edge_correct = 'All', 'All' , 4,'pubs', True
# ind_features = ['male_prop', 'income','population']
def real_Data(n_stores, n_individuals, ind_features,edge_correct,truncate ):

    """
    n_individuals -  Number of individuals
    n_features - Number of features
    n_stores -  Number of stores
    s_features -  Number of  store features
    locations_N - individuals locations
    locations_S - Store locations
    sigma_y_actual - actual error term
    truncate - Radius of the truncated Gaussian
    """

    n_stores = n_stores
    n_individuals = n_individuals

    ##Loading data
    data_source = "../Data/"

    ####-----Creating CUSTOMER level data-----
    data_resi = pd.read_csv(data_source+"london_postcode_lsoa_data.csv")

    #customers features
    data_features = pd.read_csv(data_source+"london_lsoa_imd.csv")
    data_features = data_features.drop(columns=['lat','lon'])

    #Join LSOA level data with Postcode data
    data_resi_features = data_features.merge(data_resi, left_on='lsoa', right_on = 'lsoa11cd')

    if n_individuals!='All':
        data_resi_features = data_resi_features.sample(n_individuals , random_state=5)

    #preprocessing the individual features
    features_sub = data_resi_features[['population', 'male_prop', 'index', 'income', 'employment',
    'education','health', 'crime','housing', 'living']]
    features_names = features_sub.columns

    scaler = preprocessing.MinMaxScaler() #.MinMaxScaler()StandardScaler()
    features_sub = scaler.fit_transform(features_sub)
    features_sub[:,2:10] = 1- features_sub[:,2:10] #taking inverse of the depriv

    features = np.float32(features_sub[:, [i in ind_features for i in features_names]])
    features = features.T

    ####-----Creating STORE level data-----
    data = pd.read_csv(data_source+"pubs_london_cords.csv")
    data = data.drop_duplicates(subset=['addresskey'], keep='first')

    #Store features
    data_store_specs  = pd.read_csv(data_source+ "pubs_london_height_area.csv")
    data_store_dist  = pd.read_csv(data_source+ "pubs_london_dists.csv")
    data_store_google  = pd.read_csv(data_source+ "pubs_google_data.csv")

    # Calculate the approx revenue - Average taken from VOA for pubs
    data['apx_revenue'] = data['rateablevalue']/0.095
    data_store_specs = data_store_specs[data_store_specs['total_area']<5000]

    # Edge correction - using the Edge_Sampling.py a dataset is created using the Gaussians
    # with different truncated radius showing the propotion in which falls in the study area

    if edge_correct:
        #Truncate - 5:25km; 4:20 Km; 3:15 Km;2 :10 Km;1: 5 Km
        edge_data = pd.read_csv(data_source+ "pubs_edge_prop.csv")
        edge_data = edge_data[['addresskey','prop_'+str(np.int(truncate))]] # Calculate with simplified polygon
        edge_data['edge_area'] = edge_data['prop_'+str(np.int(truncate))]
        data = data.merge(edge_data, on = 'addresskey')
        data['apx_revenue'] = np.round(data.apx_revenue * data.edge_area)
        del edge_data

    data_store_features = data_store_specs.merge(data_store_dist, on='addresskey')
    data_store_features = data_store_features.merge(data_store_google, on='addresskey')
    data_store_features = data_store_features.drop_duplicates(subset=['addresskey'], keep='first')

    #preprocessing store features
    data_sub1 = data.merge(data_store_features, on='addresskey')

    if n_stores!='All':
        data_sub1 = data_sub1.sample(n_stores,random_state=155)

    store_features = data_sub1[['sqm','height_m','total_area','metro_dist', 'rail_dist', 'bus_dist', 'parks_dist',
       'monument_dist', 'sports_dist','rating', 'user_ratings_total','inTown']].values #,,'Period' 'cnt_0_3_y', 'cnt_0_5_y','rating', 'types', 'user_ratings_total',Period

    store_features = np.float32(scaler.fit_transform(store_features))

    # poly = preprocessing.PolynomialFeatures(2,interaction_only=True, include_bias=True)
    # store_features = poly.fit_transform(store_features)

    sub_store_features = store_features
    poly = preprocessing.PolynomialFeatures(2,interaction_only=True, include_bias=False)
    store_features = poly.fit_transform(sub_store_features)
    store_features = np.float32(np.concatenate((store_features,np.repeat(1.,len(store_features)).reshape(-1,1)),axis = 1))


    data_sub2 = data_sub1[['addresskey','lat','lon', 'apx_revenue']]

    # Use approx monthly revsenue as the target variable
    Y = np.reshape(np.float32(data_sub2['apx_revenue'].values),(-1,1))
    Y = Y/12

    # log transformation
    Y = np.log(Y)

    lon_s = np.float32(data_sub2[['lon']].values)
    lat_s = np.float32(data_sub2[['lat']].values)
    locations_S = np.float32(np.concatenate((lon_s, lat_s),axis=1))

    lon_N = np.float32(data_resi_features[['lon']].values)
    lat_N = np.float32(data_resi_features[['lat']].values)
    locations_N = np.float32(np.concatenate((lon_N, lat_N),axis=1))


    ##Normalizing the coordinates
    locations = np.concatenate((locations_N, locations_S),axis=0)

    loc_min = np.transpose(np.min(locations, axis = 0)[:,np.newaxis])
    loc_max = np.transpose(np.max(locations, axis = 0)[:,np.newaxis])

    locations_S = (locations_S - loc_min)/(loc_max - loc_min)
    locations_N = (locations_N - loc_min)/(loc_max - loc_min)
    # Keeping between 1 and 10
    locations_N,locations_S = locations_N*10, locations_S*10

    return(Y,locations_N,locations_S,features,store_features)
