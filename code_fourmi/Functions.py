import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# With constraint: x_i in [-5, 5]
def function_0(n, x):
    result = 0.0
    for i in range(n):
        result += x[i] * x[i]

    return result

# With constraint: x_i in [-5, 5]
def function_1(A, n, x):
    result = A * n
    for i in range(n):
        result += (x[i] * x[i] - A * math.cos(2 * math.pi * x[i]))

    return result

def test():
    x = []
    for i in range(8):
        x.append(i-4)

    print(function_1(10, 8, x))

# With constraint: x, y in [-5, 5]
def function_1_tracer(A, n, x, y):
    results_all = []
    for i in range(101):
        results = []
        for j in range(101):
            result = A * n + (x[i][j] * x[i][j] - A * math.cos(2 * math.pi * x[i][j])) + (y[i][j] * y[i][j] - A * math.cos(2 * math.pi * y[i][j]))
            results.append(result)
        results_all.append(results)
    results_all = np.matrix(results_all)
    return results_all

def tracer():
    fig1 = plt.figure()  # create a object of drawing
    ax = Axes3D(fig1)  # use this object to create a 3d object of drawing
    X = np.arange(-5, 5.1, 0.1)
    Y = np.arange(-5, 5.1, 0.1)

    # X and Y for all possible points of axes
    X, Y = np.meshgrid(X, Y)
    Z = function_1_tracer(10, 2, X, Y)

    plt.title("Function 1")  # title
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)  # use points to draw the surface
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # the three axes with colors
    plt.show()  # show the figure of function







if __name__ == '__main__':
    tracer()