import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVR , LinearSVR
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection  import GridSearchCV ,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from Lib.rssi_gen import get_rssi

# loading the Wireless Seneor Network Data

X = np.load('../WSN/rssi.npy')
Y = np.load('../WSN/pos.npy')

# Scaling 
std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)

minmax_scale = preprocessing.MinMaxScaler().fit(X)
X_mm = minmax_scale.transform(X)


# hyperparams tuning GridSearch std(c=3000, gamma=0.003, cv=3)
"""
C_range = np.linspace(1,10000,25)
gamma_range = np.logspace(-3,3,13)
epsilon_range = np.logspace(-3,3,25)

grid_params = dict(epsilon=epsilon_range)

grid_svr = GridSearchCV(SVR(kernel='rbf',C=3000,gamma=0.003,epsilon= 1.0),param_grid=grid_params ,scoring='neg_mean_absolute_error',n_jobs = 3 ,cv =3,verbose = True)
grid_svr.fit(X , Y[:,0])

print ("The best parameters are %s with a score of %0.2f" %(grid_svr.best_params_,grid_svr.best_score_))
"""

# hyperparams tuning Randomized Search (mm: C=1600 , gamma=0.19 , cv=3 , score = -8.72 )
"""
C_range = sp_randint(30,3000)
gamma_range = uniform(0.00001,3)
dist_params = dict(C=C_range , gamma=gamma_range)

rand_svr = RandomizedSearchCV(SVR(kernel='rbf',epsilon=1.0) ,param_distributions=dist_params,n_iter=50,scoring='neg_mean_absolute_error',cv=3,n_jobs=3,
								verbose=1)
rand_svr.fit(X_mm,Y[:,0])

print ("The best parameters are %s with a score of %0.2f" %(rand_svr.best_params_,rand_svr.best_score_))
"""
"""
Xt ,Xval ,Yt , Yval = train_test_split(X_std , Y  , train_size = 0.3 , random_state =42)
Y1t = Yt[:,0]
Y2t = Yt[:,1]

svr1 = SVR(kernel='rbf', C=3000 , gamma=0.003 , epsilon=1 ) 
svr2 = SVR(kernel='rbf', C=3000 , gamma=0.003 , epsilon=1 ) 
svr1.fit(Xt,Y1t)
svr2.fit(Xt,Y2t)

pred_y1 = np.matrix ( svr1.predict(Xval) )
pred_y2 = np.matrix ( svr2.predict(Xval)  )
pred_y = np.append(pred_y1,pred_y2,axis=0).T

print "The error is ",mean_absolute_error(Yval,pred_y)
"""

# Training and Saving the best Model

svr1 = SVR(kernel='rbf', C=3000 , gamma=0.003 ) 
svr2 = SVR(kernel='rbf', C=3000 , gamma=0.003 ) 
svr1.fit(X_std,Y[:,0])
svr2.fit(X_std,Y[:,1])

# pipelining scaleing and regression
estimator_1 = [ ( 'scale' ,std_scale  ),('SVM' ,svr1 ) ]
estimator_2 = [ ( 'scale' ,std_scale  ),('SVM' ,svr2 ) ]
model_1 = Pipeline(estimator_1)
model_2 = Pipeline(estimator_2)

#_ = joblib.dump(model_1 , 'model_1.pkl')
#_ = joblib.dump(model_1 , 'model_2.pkl')




# Loading the Trajectrories
tra1 = np.load('../trajectory/tra1.npy')

#Getting the RSSIs of the the Trajectories throw okumura-hata model
X1 = get_rssi(tra1)


h1 = np.matrix ( model_1.predict(X1) )
h2 = np.matrix ( model_2.predict(X1) )
pred_tra1 = np.append(h1,h2,axis=0).T

print ("The error of the predicting the first trajectory %0.4f " %(mean_absolute_error(tra1,pred_tra1)))


plt.plot(tra1[:,0],tra1[:,1],c="g")
plt.plot(pred_tra1[:,0],pred_tra1[:,1],c="b")
plt.show()



# Loading the Second Trajectrory 
tra2 = np.load('../trajectory/tra2.npy')

#Getting the RSSIs of the the Trajectories throw okumura-hata model
X2 = get_rssi(tra2)


h1 = np.matrix ( model_1.predict(X2) )
h2 = np.matrix ( model_2.predict(X2) )
pred_tra2 = np.append(h1,h2,axis=0).T

print ("The error of the predicting the first trajectory %0.4f " %(mean_absolute_error(tra2,pred_tra2)))

plt.plot(tra2[:,0],tra2[:,1],c="g")
plt.plot(pred_tra2[:,0],pred_tra2[:,1],c="b")
plt.show()

# Loading the Third Trajectrory 
tra3 = np.load('../trajectory/tra3.npy')

#Getting the RSSIs of the the Trajectories throw okumura-hata model
X3 = get_rssi(tra3)


h1 = np.matrix ( model_1.predict(X3) )
h2 = np.matrix ( model_2.predict(X3) )
pred_tra3 = np.append(h1,h2,axis=0).T

print ("The error of the predicting the first trajectory %0.4f " %(mean_absolute_error(tra3,pred_tra3)))

plt.plot(tra3[:,0],tra3[:,1],c="g")
plt.plot(pred_tra3[:,0],pred_tra3[:,1],c="b")
plt.show()




