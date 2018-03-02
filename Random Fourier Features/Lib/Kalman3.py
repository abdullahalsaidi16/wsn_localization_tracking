import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib







def Kalman(h,R,acc_noise,init_pos= None,v=[0,0],dt=1.):
	
	# inilize lists to ploting 
	poshat = np.zeros(h.shape)


	acceleration_noise = acc_noise
	#Transition matrix    x		x'		x''		y		y'		y''     
	a = np.matrix([      [1  ,  dt  ,(dt**2)/2, 0   ,   0  ,    0 ] 
	                    ,[0  ,  1   ,  dt   ,   0 ,     0 ,     0]
		                ,[0  ,  0   ,   1   ,   0 ,     0 ,     0],
		                 [0  ,  0   ,   0,      1,      dt,  (dt**2)/2.],
		                 [0  ,  0   ,   0,      0,       1,     dt],
		                 [0  ,  0   ,   0,      0,       0,      1]])
		                        
	#Control input                a'x       a'y
	b = np.matrix([          [ (dt**3)/6.,   0      ],
							 [ (dt**2)/2.,   0      ],
							 [    dt     ,   0      ],
							 [     0     ,(dt**3)/6.],
							 [     0     ,(dt**2)/2.],
							 [     0     ,    dt    ]
							 ])
							 
	#Kalman Measurement	(get x and y )					  
	c = np.matrix         ([[1,0,0,0,0,0],
							[0,0,0,1,0,0]])
	 
	#initial estimate            x		     x'	    x''		     y		     y'		y'' 
	xhat = np.matrix(  [   [init_pos[0]],   [v[0]],   [0.],   [init_pos[1]],   [v[1]] ,  [0]]) # initial estimate
	
	
	Sz = R # measurement error covariance 
	Sw = acceleration_noise**2 * np.matrix([[ 5/35.*dt**4 , 0,  0,         0  , 0 ,0   ],
										   [ 0 ,   5/35.*dt**4,    0,         0  ,0,0   ],
										   [ 0 ,    0 ,    5/35.*dt**4,         0  ,0,0   ],
										   [ 0 ,    0 ,     0,   5/35.*dt**4   ,0,0   ],
										   [       0,         0,   0 ,0,5/35.*dt**4,0],
										   [       0,         0,    0,0,0,5/35.*dt**4]])
	P = Sw # inital estimation covariance
	
	if init_pos.all() == None :
		init_pos = h[0]
		i=0
	else:
		i= 1
		poshat[0]=(c*xhat ).T

	
	i=0
	while i < len(h):
		
		u=np.matrix([[0.0],[0.0]]) # constant acceletion command 
		

									  
		y = h[i].transpose() 
		
		# Extrapolate the most recenet state estimate to the  present time 
		
		xhat = a*xhat + b*u 
		
		#from the innovation vector 
		Inn = y- c*xhat
		
		#Compute the covarince of the innovation 
		s = c * P * (c.transpose() ) * Sz 
		
		
		# Compute Kalman Gain 
		K = a * P *( c.transpose()  )*  inv(s)
		
		#update state estimate 
		
		xhat = xhat + K*Inn
		
		#Compute the covariance matrix of ?Pos estimation error
		P = a*P*(a.transpose()) - a*P*(c.transpose()) * inv(s) * c * P * (a.transpose()) + Sw 
		
		
		
		# Save array to plot 

		poshat[i]=(c*xhat ).transpose()
	
		i+=1
		
	
	return poshat


