def count_paths(m,n,blocks):
    def count_path_in(start,blocks):
        if start[0]+1 == m and start[1]==n:
            return 1
        elif start[1]+1==n and start[0]==m:
            return 1
        if start[0]<m and (start[0],start[1]-1) not in blocks:
            count_b = count_path_in((start[0]+1,start[1]),blocks)
        else:
            count_b = 0
        if start[1]<n and (start[0]-1,start[1]) not in blocks:
            count_r = count_path_in((start[0] , start[1]+1), blocks)
        else:
            count_r=0
        return count_b+count_r
    return count_path_in((1,1),blocks)

print(count_paths(3,4,[(0,3),(1,1)]))