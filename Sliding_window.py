def slide_window(x,width,increment):
    '''
    This function takes the window size and window increment as input and produce a sequence of overlapping list from the input list
    :param x: A list of number
    :param width: An integer
    :param increment: An integer
    :return: A sequence of overlapping list
    '''
    assert increment>0 and width>0
    assert isinstance(x,list) and isinstance(increment,int) and isinstance(width,int)
    if width >= len(x):
        return [x]
    num = 0
    res=[]
    while num+width<=len(x):
        res.append(x[num:num+width])
        num+=increment
        if num+width>len(x):
            break
    return res
