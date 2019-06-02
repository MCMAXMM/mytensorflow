import numpy as np
import matplotlib.pyplot as plt
a=list()
for pos in range(70):
    i=np.arange(0,128,1)
    pe_sin=np.sin(pos/(np.power(10000,2*i/512))).reshape((128,1))
    pe_cos=np.cos(pos/(np.power(10000,2*i/512))).reshape((128,1))
    a.append(np.concatenate((pe_cos,pe_sin),axis=0))
b=np.concatenate(a,axis=1)
plt.matshow(b)
plt.show()
