def write_chunks_of_five(words,fname):
    f=open(fname,'w')
    lines=words
    num=1
    count=0
    total=len(lines)//5
    left=len(lines)%5
    for i in range(len(lines)):
        if count==total:
            for j in range(left-1):
                f.write(lines[j]+' ')
            f.write(lines[-1])
        else:
            if num%5:
                f.write(lines[i]+' ')
            else:
                f.write(lines[i])
                f.write('\n')
                count+=1
        num+=1
    return
