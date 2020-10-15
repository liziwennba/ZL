def get_power_of3(num):
    '''
    Description:This function construct any number from 1 to 40 using a set {1,3,9,27}
    Parameters: num:num is a int from 1 to 40
    '''
    assert type(num)==int
    assert num>=1
    assert num<=40
    res=[0,0,0,0]
    res[3]=num//27
    num=num%27
    res[2]=num//9
    num=num%9
    res[1]=num//3
    num=num%3
    res[0]=num//1
    print(res)
    for i in range(4):
        if res[i]==2:
            for j in range(i+1,4):
                if res[j]==2:
                    res[j]=0
                else:
                    res[i]=-1
                    res[j]+=1
                    break

    return res