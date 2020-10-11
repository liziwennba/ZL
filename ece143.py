def get_power_of3(num):
    res=[0,0,0,0]
    res[3]=num//27
    num=num%27
    res[2]=num//9
    num=num%9
    res[1]=num//3
    num=num%3
    res[0]=num//1
    for i in range(4):
        if res[i]==2:
            for j in range(i+1,4):
                if res[j]==2:
                    res[j]=0
                if res[j]==0:
                    res[i]=-1
                    res[j]=1
                    break

    return res