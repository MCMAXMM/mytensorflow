import numpy as np
#demension
import matplotlib.pyplot as plt
dims=np.arange(0,256)
positions=np.arange(0,20)
def com_encod(pos,i):
    x=np.sin(position/(np.power(10000.,2.*i/512.)))
    y=np.cos(position/(np.power(10000.,2.*i/512.)))
    return x,y
d1=np.zeros([20,256])
d2=np.zeros([20,256])
for position in positions:
    for dim in dims:
        x,y=com_encod(position,dim)
        d1[position][dim]=x
        d2[position][dim]=y
d=np.concatenate((d1,d2),axis=1)
plt.matshow(d,fignum=40)
plt.show()
