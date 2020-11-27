class Polynomial():
    '''
    Univariate polynomial
    '''
    def __init__(self,a):
        for i in a:
            assert isinstance(i,int) and isinstance(a[i],int)
        zero = True
        for i in a:
            if a[i]!=0:
                zero = False
        if not zero:
            self.a = a
            max = 0

            for i in self.a:
                if max<i:
                    max=i
            while(self.a[max]==0):
                self.a.pop(max)
                if max>0:
                    max-=1
                else:
                    break
            for i in range(max):
                if i not in self.a:
                    self.a[i]=0
        else:
            self.a = {0:0}
    def __repr__(self):
        res=''
        order = []
        temp=self.a.copy()
        remove_list=[]
        for i in temp:
            if temp[i]==0:
                remove_list.append(i)
        for i in remove_list:
            temp.pop(i)
        for i in temp:
            order.append(i)
        order = sorted(order)
        for i in order:
            if i ==0:
                if self.a[i]>0:
                    res+=str(self.a[i])
                else:
                    res+='- '+str(abs(self.a[i]))
            elif i ==1:
                if self.a[i] > 0:
                    if len(order)>1:
                        if self.a[i]==1:
                            res += ' + ' + 'x'
                        else:
                            res+=' + '+str(self.a[i])+' x'
                    else:
                        if self.a[i]==1:
                            res += 'x'
                        else:
                            res+=str(self.a[i])+' x'
                else:
                    if self.a[i]==-1:
                        res += ' - ' + 'x'
                    else:
                        res += ' - ' +str(abs(self.a[i])) + ' x'
            else:
                if self.a[i] > 0:
                    if len(order)>1:
                        if self.a[i]==1:
                            res += ' + ' + 'x^('+str(i)+')'
                        else:
                            res+=' + '+str(self.a[i]) + ' x^('+str(i)+')'
                    else:
                        if self.a[i]==1:
                            res +=  'x^('+str(i)+')'
                        else:
                            res+=str(self.a[i]) + ' x^('+str(i)+')'
                else:
                    if self.a[i]==-1:
                        res += ' - ' + 'x^('+str(i)+')'
                    else:
                        res += ' - '+str(abs(self.a[i])) + ' x^(' + str(i) + ')'

        res=res.rstrip(' + ')
        res=res.rstrip(' - ')
        return res
    def __mul__(self, other):
        temp = self.a.copy()
        if isinstance(other,int):
            for i in temp:
                temp[i]*=other
            return Polynomial(temp)
        other_temp = other.a.copy()
        res = {}
        for i in other_temp:
            for j in temp:
                p = i+j
                c = other_temp[i]*temp[j]
                if p in res:
                    res[p]+=c
                else:
                    res[p]=c
        #print('hhh',res)
        return Polynomial(res)
    def __rmul__(self, other):
        temp = self.a.copy()
        if isinstance(other,int):
            for i in temp:
                temp[i]*=other
            return Polynomial(temp)
        other_temp = other.a.copy()
        res = {}
        for i in other_temp:
            for j in temp:
                p = i+j
                c = other_temp[i]*temp[j]
                if p in res:
                    res[p]+=c
                else:
                    res[p]=c
        return Polynomial(res)
    def __add__(self, other):
        temp = self.a.copy()

        if isinstance(other,int):
            if 0 in temp:
                temp[0]+=other
            else:
                temp[0]=other
            return Polynomial(temp)
        other_temp = other.a.copy()
        for i in other.a:
            if i in temp:
                temp[i]+=other_temp[i]
            else:
                temp[i]=other_temp[i]
        return Polynomial(temp)
    def __radd__(self, other):
        temp = self.a.copy()
        if isinstance(other,int):
            if 0 in temp:
                temp[0]+=other
            else:
                temp[0]=other
            return Polynomial(temp)
        other_temp = other.a.copy()
        for i in other_temp:
            if i in temp:
                temp[i]+=other_temp[i]
            else:
                temp[i]=other_temp[i]
        return Polynomial(temp)
    def __rsub__(self, other):
        temp = self.a.copy()
        if isinstance(other,int):
            if 0 in temp:
                temp[0]=other-temp[0]
            else:
                temp[0]=other
            for i in temp:
                if i != 0:
                    temp[i] = -temp[i]
            return Polynomial(temp)
        other_temp = other.a.copy()
        for i in other_temp:
            if i in temp:
                temp[i]=other_temp[i]-temp[i]
            else:
                temp[i]=other_temp[i]
        return Polynomial(temp)
    def __sub__(self, other):
        temp=self.a.copy()
        if isinstance(other,int):
            if 0 in temp:
                temp[0]-=other
            else:
                temp[0]=-other
            return Polynomial(temp)
        other_temp = other.a.copy()
        for i in other.a:
            if i in temp:
                temp[i]-=other_temp[i]
            else:
                temp[i]=-other_temp[i]
        return Polynomial(temp)
    def subs(self,num):
        res=0
        for i in self.a:
            res += self.a[i]*(num**i)
        return res
    def __eq__(self, other):
        if isinstance(other,int):
            if other==0:
                res = True
                for i in self.a:
                    if i:
                        res=False
                return res
            elif len(self.a)==1 and 0 in self.a:
                if self.a[0]==other:
                    return True
                return False
            return False
        for i in other.a:
            if i in self.a:
                if other.a[i]!=self.a[i]:
                    return False
            elif other.a[i]!=0:
                return False
        return True
    def __rtruediv__(self, other):
        T = self.a.copy()
        for i in self.a:
            if other%self.a[i]!=0:
                raise NotImplementedError()
            else:
                T[i] = int(other/self.a[i])
        return Polynomial(T)
    def __truediv__(self, other):
        if other==0:
            raise NotImplementedError()
        if isinstance(other,int):
            T = self.a.copy()
            for i in self.a:
                if self.a[i] % other != 0:
                    raise NotImplementedError()
                else:
                    T[i] = int(self.a[i]/other)
            return Polynomial(T)
        else:
            if len(self.a) < len(other.a): raise  NotImplementedError()
            d = len(self.a) - len(other.a)
            T = self.a.copy()
            R = []
            for i in range(d + 1):
                order = []
                for j in T:
                    order.append(j)
                order = sorted(order)
                order_o = []
                for j in other.a:
                    order_o.append(j)
                order_o = sorted(order_o)

                res = True
                for nn in T:
                    if T[nn]!=0:
                        res = False
                if res:
                    R = [0]+R
                    continue
                #print(T)
                n = T[order[len(order)-1]] / other.a[order[len(order_o) - 1]]
                #print(n)
                R = [n] + R
                T1 = [0] * (d - i) + [n]
                #print(T1)
                poly_temp={}
                for j in range(len(T1)):
                    poly_temp[j]=int(T1[j])
                #print(poly_temp)
                T2 = Polynomial(poly_temp) *other

                #print(T2)
                T = (Polynomial(T)-T2).a.copy()
                #print(T)
            res= True
            for i in T:
                if T[i]!=0:
                    res = False
            if not res:
                raise NotImplementedError()
            res={}
            for i in range(len(R)):
                res[i]=int(R[i])
            #print(res)
            return Polynomial(res)



#
p=Polynomial({0:8,1:2,3:4})
q=Polynomial({0:8,1:2,2:8,4:4})
print(p*4 + 5 - 3*p - 1)
print(p.subs(10))
print(p==q)
print((p-p)==0)
print(p*q)
print(3*p)
p = Polynomial({3:-1,2:-1,0:1})
q = Polynomial({2:1,1:5})
p = Polynomial({2:1,0:-1})
q = Polynomial({1:1,0:-1})
print(p/q)
print((Polynomial({2:1})/Polynomial({1:1})==Polynomial({1: 1})))
p3=Polynomial({3:6,2:28,1:46,0:24})
q3=Polynomial({1:2,0:3})
print(p3/q3)
