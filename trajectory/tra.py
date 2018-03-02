import numpy as np
import matplotlib.pyplot as plt



# assubling trajectory for kalman
def assemble(X2,Y2):
	pos=np.zeros([len(X2),2])
	for i in range(len(X2)):
		pos[i]=np.matrix([[X2[i],Y2[i]]])
	return pos


#first trajecory
X1 = np.arange(50,150,1)
Y1 = np.arange(50,150,1)
pos1=assemble(X1,Y1)

# second trajectory
X2=np.arange(0,200,1)
Y2=-np.exp(0.0266*X2) +200
pos2 = assemble(X2,Y2)



#     ***** Theird trajectory  *****
# 1 branch
x1 = np.arange(0,220,1)
y1 = np.arange(200,180,-.09)

# branch 2
x2 = np.arange(220,-10,-1)
y2 = np.arange(179,129,-.2173)

# branch 3 
dy = .5
x3 = np.arange(-10,200,1)
y3 = np.arange(128,0,-.609)


x1= np.append(x1,x2)
x = np.append(x1,x3)

y1 = np.append(y1,y2)
y = np.append(y1,y3)

path = np.zeros((len(x),2))
for i in range(len(x)):
	path[i] = np.matrix([x[i],y[i]])

# function to smooth the third trajectory
def smooth(path,data_weight=0.3 ,smooth_weight=0.15, tolerance = 0.01):
	newpath = path
	change = tolerance
	c =0
	while c < 1000 :
		change =0.
		for i in range(1,len(path)-1):
			if not(i ==99 or i==100 or i==199 or i==200 or i==299):
				for j in range(len(path[0])):
					
					aux = newpath[i][j]
					
					newpath[i][j] += data_weight * (path[i][j]-newpath[i][j])
					
					newpath[i][j] += smooth_weight * ( newpath[i-1][j] + newpath[i+1][j] - 2.*newpath[i][j] )
					
					change += abs(aux - newpath[i][j])
			else :
				for j in range(len(path[0])):
					
					aux = newpath[i][j]
					
					newpath[i][j] += 0.8 * (path[i][j]-newpath[i][j])
					
					newpath[i][j] += 0.1 * ( newpath[i-1][j] + newpath[i+1][j] - 2.*newpath[i][j] )
					
					change += abs(aux - newpath[i][j])
					change += 0.005
		c+=1			
	return newpath

newpath = smooth(path)

np.save('tra1.npy',pos1)

plt.plot(pos2[:,0],pos2[:,1])
plt.show()

