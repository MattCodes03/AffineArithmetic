from taylor.autodiff import Variable, DifferentiableFunction

# Define the function


def f(x):
    return x**3 + 2*x**2 + x + 1


F = DifferentiableFunction(f)

# Taylor expansion point
a = 1.0
x_var = Variable(a)

# Compute derivatives at a
derivs = F.derivatives(x_var, 4)
F.print_derivatives(x_var, 4)


x = Variable(2.1)

# Compute Taylor polynomial at x
approx = F.taylor_polynomial(x, a, derivs)
exact = f(x).value

print("Taylor approx:", approx)
print("Exact value   :", exact)
