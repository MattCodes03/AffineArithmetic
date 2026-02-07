"""
------------------------------
cre: Feb 2026

web: github.com/mattcodes03
org: Univerity of Strathclyde

MIT License
------------------------------

"""
from intervals import Interval


class AffineForm:
    _noise_counter = 0  # global source of uncertainty

    @staticmethod
    def new_noise():
        AffineForm._noise_counter += 1
        return AffineForm._noise_counter

    def __init__(self, x0: float, coeffs: dict[int, float] | None = None):
        self.x0 = float(x0)
        self.coeffs = coeffs.copy() if coeffs else {}

    @classmethod
    def from_interval(cls, interval: Interval):
        a = interval.lo
        b = interval.hi

        x0 = 0.5 * (a + b)
        r = 0.5 * (b - a)

        if r == 0:
            return cls(x0)

        eps = cls.new_noise()
        return cls(x0, {eps: r})

    def __neg__(self):
        coeffs = {eps: -v for eps, v in self.coeffs.items()}
        return AffineForm(-self.x0, coeffs)

    # Total deviation
    @property
    def radius(self):
        return sum(abs(v) for v in self.coeffs.values())

    def to_interval(self):
        r = self.radius
        return Interval(self.x0 - r, self.x0 + r)

    def __repr__(self):
        terms = [f"{coef} eps{idx}" for idx, coef in self.coeffs.items()]
        if terms:
            return f"{self.x0} + " + " + ".join(terms)
        return f"{self.x0}"

    def __add__(self, other):
        if not isinstance(other, AffineForm):
            return NotImplemented

        x0 = self.x0 + other.x0
        coeffs = self.coeffs.copy()

        for eps, val in other.coeffs.items():
            coeffs[eps] = coeffs.get(eps, 0.0) + val

        return AffineForm(x0, coeffs)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, AffineForm):
            return NotImplemented

        # New centre
        x0 = self.x0 * other.x0

        coeffs = {}

        for eps, xi in self.coeffs.items():
            coeffs[eps] = coeffs.get(eps, 0.0) + xi * other.x0

        for eps, yi in other.coeffs.items():
            coeffs[eps] = coeffs.get(eps, 0.0) + yi * self.x0

        # Nonlinear remainder -> new noise symbol
        remainder = 0.0
        for xi in self.coeffs.values():
            for yi in other.coeffs.values():
                remainder += abs(xi * yi)

        if remainder != 0.0:
            new_eps = AffineForm.new_noise()
            coeffs[new_eps] = remainder

        return AffineForm(x0, coeffs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            coeffs = {eps: v/other for eps, v in self.coeffs.items()}
            return AffineForm(self.x0/other, coeffs)

        if not isinstance(other, AffineForm):
            return NotImplemented

        y0 = other.x0
        if y0 == 0:
            raise ZeroDivisionError(
                "Affine form division by interval containing zero is undefined")

        x0 = self.x0 / y0
        coeffs = {}

        # combine all noise symbols from both operands
        all_eps = set(self.coeffs.keys()).union(other.coeffs.keys())

        # linear part
        for eps in all_eps:
            xi = self.coeffs.get(eps, 0.0)
            yi = other.coeffs.get(eps, 0.0)
            coeffs[eps] = (xi * y0 - self.x0 * yi) / (y0**2)

        # nonlinear remainder (overestimation)
        remainder = sum(abs((xi*yi)/(y0**2)) for xi in self.coeffs.values()
                        for yi in other.coeffs.values())
        if remainder != 0.0:
            eps = AffineForm.new_noise()
            coeffs[eps] = remainder

        return AffineForm(x0, coeffs)

    # ---- Right-hand division by scalar ----
    def __rtruediv__(self, other):
        # other / self
        if isinstance(other, (int, float)):
            return AffineForm(other) / self
        return NotImplemented
