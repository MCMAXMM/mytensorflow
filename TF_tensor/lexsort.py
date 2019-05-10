import numpy as np
a = [1,5,1,4,3,4,4] # First column
b = [9,4,0,4,0,2,1] # Second column
#np.lexsort就是按照primary key（最后一个,axis=-1,a）来排序，如果主key中有相同的元素就按照前面的元素来排序
#最后返回index
ind = np.lexsort((b,a)) # Sort by a, then by b
print(ind)
