# This script calculate the propotion of the Gaussian on a store overalps with the study region

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import pyproj

import geopandas as gpd
import tensorflow_probability as tfp

data_source = "Data/"

#Loading pubs cordinates for london
data = pd.read_csv(data_source+ "pubs_london_cords.csv")
data = data.drop_duplicates(subset=['addresskey'], keep='first')

# Loading london boundary
london_shape = gpd.GeoDataFrame.from_file(data_source+"London_boundary/london.shp")
geom1 = london_shape.geometry[0]


#This is a mapping between 4326 cordinates system and Km
bufrs_km = {0.225:'25km',0.18:'20Km', 0.13:'15Km', 0.088:'10Km', 0.072:'5Km'}


# Radius to run the edge sampling
buf_r = 0.225 #5 - buf_r = 0.225 (25km); Truncate - 4, buf_r-0.18 (20 Km); Truncate - 3, buf_r-0.13 (15 Km); Truncate - 2, buf_r- 0.088 (10 Km); buf_r- 0.072 (5 Km);


data_sub12 = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat))
data_sub12['geometry'] = data_sub12.geometry.buffer(buf_r)

edge_data = []
for k in range(len(data_sub12)):
    data_sub13 = data_sub12.iloc[[k]]
    addresskey = data_sub12['addresskey'].iloc[k]

    geom2 = data_sub12['geometry'].iloc[k]
    poly_intersect= geom1.intersection(geom2) # multipoint
    sigma = (buf_r/4) #normal dis 4*sigma covers over 0.99

    mvn = tfp.distributions.MultivariateNormalDiag(loc=[data_sub13.iloc[0]['lon'], data_sub13.iloc[0]['lat']],scale_diag=[sigma, sigma])
    n_sampl = 100000
    smplz = mvn.sample(n_sampl).numpy()
    insmpl, insmpl_trnc = 0 , 0

    points = gpd.GeoSeries(gpd.points_from_xy(smplz[:,0],smplz[:,1],crs = "EPSG:4326"))
    insmpl_trnc += np.sum(points.intersects(geom2))
    insmpl +=  np.sum(np.sum(points.intersects(poly_intersect)))
    prop = insmpl/insmpl_trnc
    col = 'prop_'+ str(buf_r)
    edge_data.append({'addresskey': addresskey, col: prop})
edge_data_df = pd.DataFrame(edge_data)
if a ==0:
    edge_data_final = edge_data_df
else:
    edge_data_final = edge_data_final.merge(edge_data_df, on = 'addresskey')
a += 1


edge_data_final.to_csv(data_source+"_edge_prop"+bufrs_km[buf_r]+".csv")
