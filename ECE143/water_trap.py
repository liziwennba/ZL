def get_trapped_water(seq):
    dp= [0 for i in range(len(seq))]
    left = 0
    right =0
    res = 0
    for i in range(len(seq)):
        if seq[i]>0:
            if seq[left]!=0 and left==0 and i <=left:
                left = i
            else:
                right = i
                if seq[right] < seq[left]:
                    dp[i] += seq[right]*(right - left-1)
                    for j in range(left+1, right):
                        dp[i]-=seq[j]
                        dp[i]-=dp[j]
                else:
                    dp[i] +=seq[left]*(right-left-1)
                    for j in range(left+1, right):
                        dp[i]-=seq[j]
                        dp[i]-=dp[j]
                    for m in range(left + 1, right+1):
                        res += dp[m]
                    left = right
    return res

print(get_trapped_water([2, 1, 2]))

