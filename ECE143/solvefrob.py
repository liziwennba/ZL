import numpy as np
def solvefrob(coefs,b):
    dp=[[[] for i in range(b)] for j in range(len(coefs))]
    for i in range(len(coefs)):
        for j in range(0,b):
            if (j+1)%coefs[i]==0:
                temp = np.zeros([len(coefs)])
                temp[i]=(j+1)/coefs[i]
                dp[i][j].append(temp)
            value = j+1
            num = 1
            while(value-coefs[i]>0):
                if i >=1:
                    for p in range(i):
                        if dp[p][value-coefs[i]-1]:
                            for item in dp[p][value-coefs[i]-1]:
                                temp = item.copy()
                                temp[i] = num
                                dp[i][j].append(temp)
                value-=coefs[i]
                num+=1
    res =[]
    for i in range(len(coefs)):
        for item in dp[i][-1]:
            temp =tuple(item.astype(int))
            res.append(temp)
    return res

print(solvefrob([1,2,3,5],10))

