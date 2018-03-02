import numpy as np
from numpy.linalg import inv

def Kalman(dt,h,R,init_pos=[0.,0.]):
	# inilize lists to ploting 
	#pos = []
	posmea = []
	poshat = np.zeros(h.shape)
	vel = []
	velhat = []
	
	# position measurement noise (feet) 
	measurement_noise = 16.
	#acceleration input noise(feet/sec^2)
	acceleration_noise = 0.0001
	
	a = np.matrix([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])   # transition matrix
	b = np.matrix([[(dt**2)/2.,0] , [dt,0],[0,(dt**2)/2.],[0,dt]]) # input matrix 
	c = np.matrix([[1,0,0,0],[0,0,1,0]]) # measurement matrix 
	xhat = np.matrix([[ init_pos[0] ],[0],[init_pos[1]],[0]]) # initial estimate
	
	
	Sz = R # measurement error covariance 
	Sw = acceleration_noise * np.matrix([[ (dt**4)/4. , (dt**3/2.),0,0],
										   [ (dt**3)/2. , dt**2 ,0,0],
										   [0,0,(dt**4)/4.,(dt**3)/2.],
										   [0,0,(dt**3)/2.,(dt**4)/4.]])
	P = Sw # inital estimation covariance
	
	
	
	i=0
	while i < len(h):
		
		u=np.matrix([[0.0],[-0.0142688092452]]) # constant acceletion command 
		
		# simulate linear sys 
		ProcessNoise = acceleration_noise *( np.matrix([[(dt**2/2.)*np.random.randn()],[dt*np.random.randn()]]) )
		
		#x = a*x  + b*u + ProcessNoise 
		
		"""
		#Simulate noisy measurement 
		MeasureN = np.random.normal(0,abs(measurement_noise),1)[0]
		MeasureNoise = np.matrix( [[MeasureN],
									  [MeasureN]] )
									  """
									  
		y = h[i].transpose() 
		
		# Extrapolate the most recenet state estimate to the  present time 
		
		xhat = a*xhat + b*u 
		
		#from the innovation vector 
		Inn = y- c*xhat
		
		#Compute the covarince of the innovation 
		s = c * P * (c.T) * Sz 
		
		
		# Compute Kalman Gain 
		K = a * P *( c.transpose()  )*  inv(s)
		
		#update state estimate 
		
		xhat = xhat + K*Inn
		
		#Compute the covariance matrix of ?Pos estimation error
		P = a*P*(a.transpose()) - a*P*(c.transpose()) * inv(s) * c * P * (a.transpose()) + Sw 
		
		
		
		# Save array to plot 
		#pos.append(x[0,0])
		#posmea.append(y[0,0])
		poshat[i]=(c*xhat ).transpose()
		#vel.append(x[1])
		#velhat.append(xhat[1,0])
		
		i+=1
		
	
	return poshat
