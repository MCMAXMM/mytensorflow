import numpy as np
a=np.array([1,3,2,4])
c=np.array([0,1,1,0])
b=np.argsort(a)#从小到大排序，并返还原来的索引
print(b)
print(c[b])
