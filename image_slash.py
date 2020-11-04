import numpy as np
import random
def gen_rand_slash(m=6, n=6, direction='back'):
    '''
    Create a random slash image
    :param m: Row
    :param n: Column
    :param direction: Back or forward
    :return: The slash image
    '''
    assert m>2 and isinstance(m,int)
    assert n>2 and isinstance(n,int)
    assert isinstance(direction,str)
    res = np.zeros((m, n))
    summ = sum(list(range(min(m-1,n-1)+1)))
    p =[]
    p.append(summ)
    if m-2>0:
        for i in range(1,m-1):
            hhh=sum(list(range((min(m-i,n)-1)+1)))
            p.append(hhh)
            summ+=(hhh)
    if n-2>0:
        for i in range(1,n-1):
            hhh = sum(list(range((min(n-i,m)-1)+1)))
            p.append(hhh)
            summ+=(hhh)
    p=np.array(p)
    p=p/summ
    a = np.random.choice(len(p),p=p.ravel())
    if a==0:
        p=[]
        summ=0
        for i in range(min(n-1, m - 1)):
            summ+=min(n-1, m - 1)-i
            p.append(min(n-1, m - 1)-i)
        p=np.array(p)
        p=p/summ
        aa=np.random.choice(len(p),p=p.ravel())
        r=random.randint(2,min(n-1,m-1)-aa+1)
        for i in range(r):
            res[aa+i,aa+i]=1
    elif a<m-1:
        p=[]
        summ=0
        for i in range(min(n-1, m-a-1)):
            summ+=min(n-1, m - a-1)-i
            p.append(min(n-1, m - a-1)-i)
        p = np.array(p)
        p = p / summ
        aa = np.random.choice(len(p), p=p.ravel())
        r=random.randint(2,min(n-1, m-a-1)-aa+1)
        for i in range(r):
            res[a+aa+i,aa+i]=1
    else:
        a=a-m+2
        p=[]
        summ=0
        for i in range(min(n-a-1, m-1)):
            summ+=min(n-1-a, m-1)-i
            p.append(min(n-a-1, m-1)-i)
        p = np.array(p)
        p = p / summ
        aa = np.random.choice(len(p), p=p.ravel())
        r=random.randint(2,min(n-a-1, m-1)-aa+1)
        for i in range(r):
            res[aa+i,a+aa+i]=1

    if direction!='back':
        for m in res:
            for j in range(n // 2):
                m[j], m[n - 1 - j] = m[n - 1 - j], m[j]

    return res
