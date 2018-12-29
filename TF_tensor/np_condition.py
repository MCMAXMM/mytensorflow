import numpy as np
b=np.array([1,2,3,4,5,6,7,8,9])
c=np.array([12,3,2,4,2,5,23,52,2])
print(b[c>5])#只查找c>5的b对应的值
#[1,7,8]
