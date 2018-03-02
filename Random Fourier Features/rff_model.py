import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR , LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_approximation import RBFSampler
from Lib.Kalman2 import Kalman

# Loading the wireless sensor network Data 
X = np.load('../WSN/rssi.npy')
Y = np.load('../WSN/pos.npy')

# Scaling 
std_scale = StandardScaler().fit(X)
X_std = std_scale.transform(X)

# Linear SVR
C_range = np.linspace(10,5000,25)
grid_params = {'svm__C':C_range}
#svr1 = GridSearchCV(LinearSVR(),param_grid = grid_params,n_jobs=2,scoring = 'neg_mean_absolute_error')



# Creating pipline from kernel approximation to linear svr
fourier_feature_map = RBFSampler(gamma=0.03)
estimator_1 =Pipeline( [('scale',std_scale),('feature_map',fourier_feature_map),('svm',LinearSVR() )] )
"""
estimator_2 = [('scale',std_scale),('feature_map',fourier_feature_map),('svm',LinearSVR() )]
rff_model_1 = GridSearchCV(Pipeline(estimator_1) , param_grid = grid_params ,  scoring = 'neg_mean_absolute_error',n_jobs=2 )
rff_model_2 = Pipeline(estimator_2)
"""
 
sample_size = np.arange(30,180,10)

for D in sample_size:
	estimator_1.set_params(feature_map__n_components = D )
	rff_model_1 = GridSearchCV(estimator_1 , param_grid = grid_params ,  scoring = 'neg_mean_absolute_error',n_jobs=2,cv=3 )
	rff_model_1.fit(X,Y[:,0])
	print ("With %d samples the best params for here is %s and best score is %0.3f " %(D,rff_model_1.best_params_,rff_model_1.best_score_) )
	
	raw_input("press any key to continue")
	
