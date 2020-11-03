import numpy as np
import random
from matplotlib.pylab import subplots, cm
def gen_rand_slash(m=6, n=6, direction='back'):
    res = np.zeros((m, n))
    if direction=='back':
        a=randomint(0,m-2+n-3)
        if a==0:
            r=randint(3,min(m,n))
            for i in r:
                res[i,i]=1
        elif a<m-2:
            rand_1=random.randint(3,min(m-a,n))
            for i in range(rand_1):
                res[rand_0+i,i]=1
        else:
            rand_0=random.randint(0,n-3)
            rand_1=random.randint(3,min(n-rand_0,m))
            res=np.zeros((m,n))
            for i in range(rand_1):
                res[i,rand_0+i]=1
    else:
        if random.randint(1,2)==1:
            rand_0=random.randint(0,m-3)
            rand_1=random.randint(3,min(m-rand_0,n))
            res=np.zeros((m,n))
            for i in range(rand_1):
                res[rand_0+i,n-i-1]=1
        else:
            rand_0=random.randint(0,n-3)
            rand_1=random.randint(3,min(n-rand_0,m))
            res=np.zeros((m,n))
            for i in range(rand_1):
                res[i,n-rand_0-i-1]=1

    return res

x=np.eye(6)
fig, ax = subplots()
ax.imshow(x,cmap=cm.gray_r)