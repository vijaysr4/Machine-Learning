#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Projects/Project_1/pr1.csv')
data_set.head()
data_set.info()

data_set_gal = data_set.loc[(data_set["class"]) == 'GALAXY']
data_set_gal.head()

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from ipykernel import kernelapp as app


radec = SkyCoord(ra = data_set_gal['ra']*u.degree, dec = data_set_gal['dec']*u.degree, frame = 'icrs')
galactic = radec.galactic
print(radec)

data_set_gal['l'] = galactic.l.value
data_set_gal['b'] = galactic.b.value

r = cosmo.comoving_distance(data_set_gal['redshift'])
data_set_gal['distance'] = r.value

data_set_gal.head()

def cartesian(dist, alpha, delta):
    x = dist*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(alpha))
    
    y = dist*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(alpha))
    
    z = dist*np.sin(np.deg2rad(delta))
    return x, y, z

cart = cartesian(data_set_gal['distance'], data_set_gal['ra'], data_set_gal['dec'])
data_set_gal['x_coord'] = cart[0]
data_set_gal['y_coord'] = cart[1]
data_set_gal['z_coord'] = cart[2]
data_set_gal.head()


#plotting galaxies in coordinates
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data_set_gal['x_coord'], data_set_gal['y_coord'], data_set_gal['z_coord'], s = 10)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Y (mpc)')
ax.set_zlabel('Z (mpc)')
ax.set_title('Galactic Distrubution from SDSS', fontsize = 18)
plt.show()


fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(111)
ax.scatter(data_set_gal['x_coord'], data_set_gal['y_coord'], s = 0.5)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Y (mpc)')
ax.set_title('Galactic Distribution from SDSS in X and Y Space', fontsize = 18)
plt.show()


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
ax.scatter(data_set_gal['x_coord'],data_set_gal['z_coord'], s = 0.5)
ax.set_xlabel('X (mpc)')
ax.set_ylabel('Z (mpc)')
ax.set_title('Galactic Distribution from SDSS in X and Z Space',fontsize = 18)
plt.show()


fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(111)
ax.scatter(data_set_gal['y_coord'],data_set_gal['z_coord'], s = 0.5)
ax.set_xlabel('Y (mpc)')
ax.set_ylabel('Z (mpc)')
ax.set_title('Galactic Distribution from SDSS in Y and Z Space', fontsize = 18)
plt.show()


import seaborn as sb

fig = plt.figure(figsize = (12, 10))
sb.distplot(data_set_gal['redshift'])
plt.title('Redshift Distribution',fontsize=18)
plt.show()

fig = plt.figure(figsize=(12, 10))
sb.distplot(data_set_gal['distance'])
plt.title('Distance Distribution (MPC)',fontsize = 18)
plt.show()


data_set_gal['redshift'].describe()

data_set_gal['distance'].describe()