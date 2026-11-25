import numpy as np 

def second_derivative(f, x, h, order=2):
    x = np.asarray(x, dtype=float)

    if order == 2:
        return (F(x + h) - 2 * f(x) + f(x - h)) / h ** 2
    
    elif order == 4:
        return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * h **2)
    else:
        raise ValueError("order must be 2 or 4")

