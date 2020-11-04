class Interval(object):
    def __init__(self,start,end):
        """
        :a: integer
        :b: integer
        """
        assert start < end
        assert isinstance(start, int)
        assert isinstance(end, int)
        self._a = start
        self._b = end

    def __repr__(self):
        '''
        Return the string of class
        '''
        return f'Interval({self._a},{self._b})'

    def __eq__(self, other):
        '''
        See if the two class are equal
        :param other: Another class
        '''
        if type(self)==type(other) and self._a==other._a and self._b==other._b:
            return True
        else:
            return False

    def __lt__(self, other):
            pass

    def __gt__(self, other):
            pass

    def __ge__(self, other):
        pass

    def __le__(self, other):
        pass

    def __add__(self, other):
        '''
        add two class
        :param other: Another class
        '''
        if other._a>=self._b or other._b<=self._a:
            return [Interval(self._a,self._b),Interval(other._a,other._b)]
        else:
            min_=min(self._a,other._a)
            max_=max(self._b,other._b)
            res=Interval(min_,max_)
            return res


