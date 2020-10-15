def write_columns(data,fname):
    assert type(data)==list
    assert type(fname)==str
    for i in data:
        assert type(i)==int or type(i)==float
    data_value_2=[x**2 for x in data]
    data_value_3=[]
    for i in range(len(data)):
        data_value_3.append((data[i]+data_value_2[i])/3)
    f=open(fname,'w')
    for i in range(len(data)):
        if type(data[i])==int:
            f.write('%d,%d,%3.2f'%(data[i],data_value_2[i],data_value_3[i]))
            f.write('\n')
        else:
            f.write('%3.2f,%3.2f,%3.2f'%(data[i],data_value_2[i],data_value_3[i]))
            f.write('\n')