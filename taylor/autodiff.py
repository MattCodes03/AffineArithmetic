import numpy as np
from collections import defaultdict


class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other): return Variable.add(self, other)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Variable(other) - self
    def __mul__(self, other): return Variable.mul(self, other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return Variable.div(self, other)
    def __rtruediv__(self, other): return Variable(other) / self
    def __neg__(self): return Variable.mul(self, Variable(-1))

    def __pow__(self, n):
        result = Variable(1)
        for _ in range(n):
            result *= self
        return result

    @staticmethod
    def add(a, b):
        if not isinstance(b, Variable):
            b = Variable(b)
        value = a.value + b.value
        grads = ((a, lambda pv: pv), (b, lambda pv: pv))
        return Variable(value, grads)

    @staticmethod
    def mul(a, b):
        if not isinstance(b, Variable):
            b = Variable(b)
        value = a.value * b.value
        grads = ((a, lambda pv: pv * b), (b, lambda pv: pv * a))
        return Variable(value, grads)

    @staticmethod
    def div(a, b):
        if not isinstance(b, Variable):
            b = Variable(b)
        value = a.value / b.value
        grads = ((a, lambda pv: pv / b), (b, lambda pv: pv * (-a / (b*b))))
        return Variable(value, grads)

    def sin(self):
        return Variable(np.sin(self.value), ((self, lambda pv: pv * self.cos()),))

    def cos(self):
        return Variable(np.cos(self.value), ((self, lambda pv: pv * (-self.sin())),))

    def exp(self):
        v = np.exp(self.value)
        return Variable(v, ((self, lambda pv: pv * v),))

    def log(self):
        return Variable(np.log(self.value), ((self, lambda pv: pv / self),))

    def gradients(self):
        grads = defaultdict(lambda: Variable(0))

        def compute(v, path_val):
            for child, loc_grad in v.local_gradients:
                value_to_child = loc_grad(path_val)
                grads[child] += value_to_child
                compute(child, value_to_child)
        compute(self, Variable(1))
        return grads


class DifferentiableFunction:
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def derivatives(self, x, n):
        y = self.func(x)
        derivs = []
        current = y
        for _ in range(n):
            grads = current.gradients()
            current = grads[x]
            derivs.append(current)
        return derivs

    def taylor_polynomial(self, x, x0, derivs):
        dx = x.value - x0
        result = self.func(Variable(x0)).value
        fact = 1
        power = 1
        for k, d in enumerate(derivs, start=1):
            fact *= k
            power *= dx
            result += d.value * power / fact
        return result

    def print_derivatives(self, x, n):
        derivs = self.derivatives(x, n)
        y0 = self.func(x).value
        print(f"f({x.value}) = {y0}")
        for i, d in enumerate(derivs, start=1):
            print(f"f^{i}({x.value}) = {d.value}")
        return derivs
