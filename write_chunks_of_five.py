def write_chunks_of_five(words,fname):
    assert type(fname)==str
    assert type(words)==list
    for i in words:
        assert type(i)==str
    f=open(fname,'w')
    words_re={}
    lines=words
    num=1
    for i in range(len(lines)):
        if lines[i] not in words_re:
            if num%5:
                if i==len(lines)-1:
                    f.write(lines[i])
                    f.write('\n')
                else:
                    f.write(lines[i]+' ')
                words_re[lines[i]]=i
            else:
                f.write(lines[i])
                words_re[lines[i]]=i
                f.write('\n')
            num+=1
    return

