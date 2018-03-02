import numpy as np
from random import *
from numpy.linalg import inv
import matplotlib.pyplot as plt

def Kalman(h,R,acc_noise,init_pos= None,v=[0,0],dt=1.):
	
	# inilize lists to ploting 
	poshat = np.zeros(h.shape)
	
	
	#acceleration input noise (feet/sec^2)
	acceleration_noise = acc_noise
	
	a = np.matrix([[1.,0],[0,1]])   # transition matrix
	b = np.matrix([[dt,0] , [0,dt]]) # input matrix 
	c = np.matrix([[1,0],[0,1]]) # meeasurement matrix 
	x = np.matrix([ [init_pos[0]],[init_pos[1] ] ]) # initial position 
	xhat = x # initial estimate
	
	
	Sz = R # measurement error covariance 
	Sw = acceleration_noise**2 * np.matrix([ [dt**2,0],[0,dt**2]])
	P = Sw # inital estimation covariance
	
	if init_pos.all() == None :
		init_pos = h[0]
		i=0
	else:
		i= 1
		poshat[0]=(c*xhat ).T
	 
	i=0
	while i <(len(h)):
		
		u= np.matrix([[v[0]],[v[1]]]) # constant acceletion command 
		

		
		#Simulate noisy measurement
		y = h[i].T
		
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
"""
duration = 50
dt = 0.1
# inilize lists to ploting 
pos = []
posmea = []
poshat = []
vel = []
velhat = []

Kalman(duration,dt)

print 

# my plotting :D
t = np.arange(0,duration,dt)
print len(pos)
# select first figure  
plt.figure(1)
plt.plot(t,pos,'r-',label='Position')
plt.plot(t,posmea,'b-',label='Position measurement')
plt.plot(t,poshat,'g-',label='Position estimation')
plt.legend()

# select the second figure 
#plt.figure(2)


#select the third figure 
#plt.figure(3)



plt.legend()
plt.show()
"""
