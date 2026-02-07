from taylor.autodiff import Variable, DifferentiableFunction


def f(x):
    return x**3 + 2*x**2 + x + 1


F = DifferentiableFunction(f)

x0 = 1.0
x_var = Variable(x0)
derivs = F.derivatives(x_var, 4)
F.print_derivatives(x_var, 4)


# Taylor approx at x=2.1
x_test = Variable(2.1)
approx = F.taylor_polynomial(x_test, x0, derivs)
exact = f(x_test).value

print("Taylor approx:", approx)
print("Exact value   :", exact)
