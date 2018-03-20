import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def WSN(x , y, anchor_points ,offline_points):
	RSSI = np.zeros( (offline_points , anchor_points) )
	pos = np.zeros( (offline_points , 2) )
	sensors = np.zeros((anchor_points,2))
	
	# Coordinates of the Anchor points (Sensors)
	X = np.linspace(12,x-12,np.sqrt(anchor_points))
	Y = np.linspace(12,y-12,np.sqrt(anchor_points))
	i=0
	for x1 in X :
		for y1 in Y :
			sensors[i] = np.array( (x1,y1) )
			i += 1
	
	# Coordinates of offline points
	X = np.linspace(0,x,np.sqrt(offline_points) )
	Y = np.linspace(0,y,np.sqrt(offline_points) )
	
	i = 0
	for x1 in X :
		for y1 in Y:
			j = 0
			for sen in sensors:
				pos[i] = np.array((x1,y1) )
				#  rssi simulation via Okumura-Hata model 
				RSSI[i,j] = 100 - 100*4 * np.log10( norm(pos[i]-sen) )
				RSSI[i,j] += np.random.normal(0 , abs(0.05*RSSI[i,j]) , 1 )
				j += 1
			i += 1 
				
	 
	return RSSI , pos , sensors
	
rssi , pos , sen = WSN(200,200,16,200)
#np.save('rssi.npy',rssi)
#np.save('pos.npy',pos)
"""
plt.scatter( sen[:,0], sen[:,1], marker="^", c="r",label="Stationary sensors" )
plt.scatter( pos[:,0], pos[:,1], marker="+", c="b",label="Offline postions")
plt.legend(loc='upper right',framealpha=1)
plt.show()

"""
