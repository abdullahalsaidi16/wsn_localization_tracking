import numpy as np


# Coordinates of the Anchor points (Sensors) 
X = np.linspace(12,200-12,np.sqrt(16))
Y = np.linspace(12,200-12,np.sqrt(16))
sensors = np.zeros( (16,2) ) 
i=0
for x1 in X :
	for y1 in Y :
		sensors[i] = np.array( (x1,y1) )
		i += 1


# okumura hata rssi simulation
def get_rssi( pos , sensor = sensors):
	X=np.zeros([len(pos),len(sensor)])
	i=0
	for p in pos:
		j=0
		for sen in sensor:
			rssi = 100-100*4*np.log10(np.linalg.norm(p-sen))
			rssi += np.random.normal(0,abs(0.05*rssi),1)
			X[i,j] = rssi
			
			j += 1
			
		i += 1
		
	return X
			
		
	
