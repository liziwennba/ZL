# def next_permutation(t:tuple)->tuple:
#     last = t[-1]
#     temp = 0
#     res=[]
#     for i in range(len(t)):
#         if t[len(t)-i-1]<last:
#             temp = len(t)-i
#             break
#     print(temp)
#     if temp ==0:
#         for i in range(len(t)):
#             res.append(t[-i-1])
#         return res
#     for i in range(temp-1):
#         res.append(t[i])
#     res.append(t[-1])
#     res.append(t[temp-1])
#     for i in range(temp+1,len(t)):
#         res.append(t[i-1])
#     return res
def next_permutation(t:tuple)->tuple:
    assert isinstance(t,tuple)
    for i in t:
        assert isinstance(i,int)
    temp = 0
    res=[]
    length=len(t)
    p=t[-1]
    a = list(t)
    for i in range(length):
        if t[length-i-1]>t[length-i-2]:
            temp=len(t)-i-1
            break
    if temp==0:
        a.reverse()
    elif temp==length-1:
        a=a[0:-2]+a[-1:-3:-1]
    else:
        last = -1
        while(t[last]<t[temp-1]):
            last-=1
        a[last],a[temp-1]=a[temp-1],a[last]
        a=a[0:temp]+a[-1:temp-1:-1]
    print(temp)
    return a
print(next_permutation((1,2,3)))