import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.svm import SVR , LinearSVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Lib.rssi_gen import get_rssi
from Lib.Kalman3 import Kalman as Kalman_3
from Lib.Kalman2 import Kalman as Kalman_2
from Lib.Kalman import Kalman as Kalman_1
from WSN.WSN import WSN

np.random.seed(76)

def predict_svm(X1):
	h1 = np.matrix ( svm1.predict(X1) )
	h2 = np.matrix ( svm2.predict(X1) )
	return np.append(h1,h2,axis=0).T

def predict_rff(X1):
	X1 =std_scale.transform(X1)
	h1 = np.matrix ( rff_svm1.predict(X1) )
	h2 = np.matrix ( rff_svm2.predict(X1) )
	return np.append(h1,h2,axis=0).T
	
def err(Y,H):
	return np.sqrt( ( Y[:,0]-H[:,0] )**2 + ( Y[:,1]-Y[:,1] )**2 )
	
	
# loading the Wireless Seneor Network Data
X = np.load('WSN/rssi.npy')
Y = np.load('WSN/pos.npy')


# Scaling 
std_scale = StandardScaler().fit(X)
X_std = std_scale.transform(X)



# Training and Saving the Model
svr1 = SVR(kernel='rbf', C=3000 , gamma=0.003 ) 
svr2 = SVR(kernel='rbf', C=3000 , gamma=0.003 ) 
svr1.fit(X_std,Y[:,0])
svr2.fit(X_std,Y[:,1])

# pipelining scaleing and regression
svm1 = Pipeline ( [ ( 'scale' ,std_scale  ),('SVM' ,svr1 ) ] )
svm2 = Pipeline ( [ ( 'scale' ,std_scale  ),('SVM' ,svr2 ) ] )

# Random Fourier Feature Pipelining
rff_svm1 = Pipeline( [  ('feature_map' , RBFSampler(n_components=90,gamma=0.05,random_state=1) ) , ('SVM',LinearSVR(C=300)) ] )
rff_svm2  = Pipeline( [  ('feature_map' , RBFSampler(n_components=90,gamma=0.05,random_state=1) ) , ('SVM',LinearSVR(C=300)) ] )
rff_svm1.fit(X_std,Y[:,0])
rff_svm2.fit(X_std,Y[:,1])



# Loading the Trajectrories
tra1 = np.load('trajectory/tra1.npy')
tra2 = np.load('trajectory/tra2.npy')
tra3 = np.load('trajectory/tra3.npy')


#Getting the RSSIs of the the Trajectories throw okumura-hata model
X1 = get_rssi(tra1)
X2 = get_rssi(tra2)
X3 = get_rssi(tra3)


#  measurent error covariance R -_-
ref_points = np.matrix( np.random.normal( 100,30,(100,2) ) )
ref_rssi = get_rssi(ref_points)

ref_pred = predict_svm(ref_rssi)
ref_err = abs(ref_points - ref_pred )
r = np.var(ref_err,0)
r= r/3.
R = np.multiply( np.matrix([[1],[1]])*r , np.matrix(np.eye(r.shape[1]))  )




#Predicting and Filtering the real Trajectory throw SVM

"""
#Trajectory 1
init_val =[1,1]
pred_tra1_svm = predict_svm(X1)
pred_tra1_rff = predict_rff(X1)

# Kalman 1st order
#poshat_1 = Kalman_1(pred_tra1_svm,R,0.00001,tra1[0],init_val)
#poshat_1_rff = Kalman_1(pred_tra1_rff,R,0.00001,tra1[0],init_val)

# Kalman 2nd order
poshat_1 = Kalman_2(pred_tra1_svm,R,0.0001,tra1[0],init_val)
poshat_1_rff = Kalman_2(pred_tra1_rff,R,0.0001,tra1[0],init_val)

# Kalman 3rd order
#poshat_3 = Kalman_3(pred_tra1,R,0.01,tra1[0],init_val)


print ("The error of the predicting the first trajectory %0.4f " %(mean_absolute_error(tra1,pred_tra1_svm)))
print ("The error of the Kalman the first trajectory %0.4f " %(mean_absolute_error(tra1,poshat_1)))
print ("The error of the RFF-Kalman the first trajectory %0.4f " %(mean_absolute_error(tra1,poshat_1_rff)))

plt.plot(tra1[:,0],tra1[:,1],"b-",label='Real trajectory')
plt.plot(pred_tra1_svm[:,0],pred_tra1_svm[:,1],"*",markersize=2.5,label='SVM Prediction',color='orange')
plt.plot(poshat_1_rff[:,0],poshat_1_rff[:,1],"g--",markersize=2,label='RFF+Kalman')
plt.plot(poshat_1[:,0],poshat_1[:,1],"r--",markersize=4,label='SVM+Kalman')
plt.legend(loc='lower right',framealpha=1)
plt.show() 



# Trajectory 2
init_val = [1.,0.]
pred_tra2_svm = predict_svm(X2)
pred_tra2_rff = predict_rff(X2)

# Kalman 1st order
#poshat_2 = Kalman_1(pred_tra2_svm,R,0.002,tra2[0],init_val)
#poshat_2_rff = Kalman_1(pred_tra2_rff,R,0.001,tra2[0],init_val)

# Kalman 2nd order
poshat_2 = Kalman_2(pred_tra2_svm,R,0.001,tra2[0],init_val)
poshat_2_rff = Kalman_2(pred_tra2_rff,R,0.001,tra2[0],init_val)

# Kalman 3rd order
#poshat_2 = Kalman_3(pred_tra2,R,0.002,tra2[0])


print ("The error of the predicting the Second trajectory %0.4f " %(mean_absolute_error(tra2,pred_tra2_svm)))
print ("The error of the Kalman the Second trajectory %0.4f " %(mean_absolute_error(tra2,poshat_2)))
print ("The error of the RFF-Kalman the Second trajectory %0.4f " %(mean_absolute_error(tra2,poshat_2_rff)))

plt.plot(tra2[:,0],tra2[:,1],"b-",label='Real trajectory')
plt.plot(pred_tra2_svm[:,0],pred_tra2_svm[:,1],"*",markersize=2.5,label='SVM Prediction',color='orange')
plt.plot(poshat_2_rff[:,0],poshat_2_rff[:,1],"g--",markersize=2,label='RFF+Kalman')
plt.plot(poshat_2[:,0],poshat_2[:,1],"r--",markersize=4,label='SVM+Kalman')
plt.legend(loc='upper right',framealpha=1)
plt.show() 

"""
# Trajectory 3
init_val=[0,0]
pred_tra3_svm = predict_svm(X3)
pred_tra3_rff = predict_rff(X3)

# Kalman 1st order
#poshat_3 = Kalman_1(pred_tra3,R,0.2,tra3[0],init_val)

# Kalman 2nd order
poshat_3 = Kalman_2(pred_tra3_svm,R,0.2,tra3[0],init_val)
poshat_3_rff = Kalman_2(pred_tra3_rff,R,0.2,tra3[0],init_val)

# Kalman 3rd order
#poshat_3 = Kalman_3(pred_tra3,R,0.001,tra3[0],init_val)

print ("The error of the predicting the Third trajectory %0.4f " %(mean_absolute_error(tra3,pred_tra3_svm)))
print ("The error of the Kalman the Third trajectory %0.4f " %(mean_absolute_error(tra3,poshat_3)))
print ("The error of the RFF-Kalman the Third trajectory %0.4f " %(mean_absolute_error(tra3,poshat_3_rff)))

plt.plot(tra3[:,0],tra3[:,1],"b-",label='Real trajectory')
plt.plot(pred_tra3_svm[:,0],pred_tra3_svm[:,1],"*",markersize=2.5,label='SVM Prediction',color='orange')
plt.plot(poshat_3_rff[:,0],poshat_3_rff[:,1],"g--",markersize=2,label='RFF+Kalman')
plt.plot(poshat_3[:,0],poshat_3[:,1],"r--",markersize=4,label='SVM+Kalman')
plt.legend(loc='upper right',framealpha=1)
plt.show() 


# Ploting error of svm and rff in trajectory 

pred_tra3_svm = predict_svm(X3)
sample_size = np.arange(20,180,10)
error = {'SVM':[mean_absolute_error(tra3,poshat_3)],'RFF':[]}
Kalman_improve = {'SVM':[mean_absolute_error(tra3,pred_tra3_svm) - mean_absolute_error(tra3,poshat_3)],'RFF':[]}

for D in sample_size:
	rff_svm1.set_params(feature_map__n_components = D , feature_map__gamma = 0.05 )
	rff_svm2.set_params(feature_map__n_components = D , feature_map__gamma = 0.05  )
	rff_svm1.fit(X_std,Y[:,0])
	rff_svm2.fit(X_std,Y[:,1])
	
	
	pred_tra3_rff = predict_rff(X3)
	
	poshat_3_rff = Kalman_2(pred_tra3_rff,R,0.2,tra3[0],init_val)
	
	error['RFF'].append(mean_absolute_error(tra3,poshat_3_rff))
	Kalman_improve['RFF'].append( mean_absolute_error(tra3,pred_tra3_rff ) - mean_absolute_error(tra3,poshat_3_rff) )



plt.ylabel('Mean absolute error')
plt.xlabel('Components of RFF')
plt.plot(sample_size,error['SVM']*len(sample_size),'b-',label="SVM + KF")
plt.plot(sample_size,error['RFF'],c='orange',label="RFF + KF")
plt.legend(loc='upper right')
plt.show()

# Plotting Kalman Imporve
#plt.ylabel('Kalman I')
plt.xlabel('Components of RFF')
plt.plot(sample_size,Kalman_improve['SVM']*len(sample_size),'b-',label="SVM + KF")
plt.plot(sample_size,Kalman_improve['RFF'],c='orange',label="RFF + KF")
plt.legend(loc='upper right')
plt.show()



# Ploting the learning curve With Kalman
training_curve = []
x = np.arange(100,1000,10)
for t in x :
	rssi , pos , sen =  WSN(200,200,16,t)
	rssi = std_scale.transform(rssi)
	
	rff_svm1.fit(rssi,pos[:,0])
	rff_svm2.fit(rssi,pos[:,1])
	
	
	pred_tra3_rff = predict_rff(X3)
	poshat_3_rff = Kalman_2(pred_tra3_rff,R,0.2,tra3[0],init_val)
	
	training_curve.append(mean_absolute_error(tra3,poshat_3_rff))

#ploting 
plt.title('RFF Learning Curves with KF')
plt.xlabel('Training Examples')
plt.ylabel('Mean absolute error')
plt.plot(x,training_curve,c='blue',label="RFF + KF")
plt.show()


"""
x = np.arange(0,len(tra3),1)	
#ploting
my_title = 'RFF with '+str(D)+' components'
plt.title( my_title )
plt.plot(x,E1,"r-",label="E1")
plt.plot(x,E2,"b-",label="E2")
plt.legend(loc='upper right',framealpha=1)
plt.show()
"""

