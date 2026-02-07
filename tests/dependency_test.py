from affine import AffineForm
from intervals import Interval

print("Affine:")
x = AffineForm.from_interval(Interval(0, 1))
print(x-x)

print("Interval:")
x = Interval(0, 1)
print(x-x)
