class Rational():
    def _gcd(m, n):
        if n == 0:
            m, n = n, m
        while m != 0:
            m, n = n % m, m
        return n

    def __init__(self, num, den):
        assert den != 0
        assert isinstance(num, int)
        assert isinstance(den, int)
        g = Rational._gcd(num, den)
        self.sign = False
        if num < 0:
            self.sign = True
        if den < 0:
            self.sign = not self.sign
        self._num = num // g
        self._den = den // g

    def __int__(self):
        return self._num // self._den

    def __float__(self):
        return self._num / self._den

    def __repr__(self):
        if abs(self._den) == 1:
            if self.sign:
                return '-' + str(abs(self._num))
            else:
                return str(abs(self._num))
        if self.sign:
            return '-' + str(abs(self._num)) + '/' + str(abs(self._den))
        return str(abs(self._num)) + '/' + str(abs(self._den))

    def __lt__(self, other):
        if self._num * other._den < self._den * other._num:
            return 1
        else:
            return 0

    def __gt__(self, other):
        if self._num * other._den < self._den * other._num:
            return 0
        else:
            return 1

    def __neg__(self):
        num = -self._num
        return Rational(num, self._den)

    def __eq__(self, another):
        return self._num * another._den == self._den * another._num

    def __mul__(self, r):
        if isinstance(r, int):
            return Rational(self._num * r, self._den)
        return Rational(self._num * r._num, self._den * r._den)

    def __add__(self, r):
        return Rational(self._num * r._den + r._num * self._den, self._den * r._den)

    def __sub__(self, r):
        return Rational(self._num * r._den - self._den * r._num, self._den * r._den)

    def __truediv__(self, r):
        if isinstance(r, int):
            return Rational(self._num, self._den * r)
        return Rational(self._num * r._den, self._den * r._num)

    def __rtruediv__(self, r):
        return Rational(self._den * r, self._num)

    def __rmul__(self, r):
        return Rational(self._num * r, self._den)

    def __cmp__(self, other):
        if self._num * other._den < self._den * other._num:
            return -1
        elif self._num * other._den == self._den * other._num:
            return 0
        else:
            return 1
