def fibonacci(n):
    '''
    Return the number of fibonacci sequence
    '''
    assert isinstance(n,int)
    assert n>=1
    a=1
    b=1
    num=0
    while num<n:
        yield a
        a,b=b,a+b
        num=num+1

