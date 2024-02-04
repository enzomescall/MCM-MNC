import pde 


def mu(x, y):
    return [x, y]

grid = pde.CartesianGrid([[-5, 5], [-10, 10]], [20, 10], periodic = [True, False])

field = pde.VectorField.from_expression(grid, ["sin(x)", "cos(x)"])
field.plot(method="streamplot", title="Stream plot")

# diffusivity = "1.01 + tanh(x)"
# term_1 = f"({diffusivity}) * laplace(c)"
# term_2 = f"dot(gradient({diffusivity}), gradient(c))"
# eq = PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})


fp = pde.PDE({"p":"- divergence(p * )"})