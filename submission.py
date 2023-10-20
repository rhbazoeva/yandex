import numpy as np
import math as mt
from numpy.linalg import norm
from typing import Callable, Tuple, Union, List


class f1:
    def __call__(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return x ** 2

    def grad(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return 2 * x

    def hess(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return 2


class f2:
    def __call__(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return mt.sin(3 * x ** (3 / 2) + 2) + x ** 2

    def grad(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return 9 / 2 * (x ** (1 / 2)) * mt.cos(3 * x ** (3 / 2) + 2) + 2 * x

    def hess(self, x: float):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return 9 / 4 * x ** (-1 / 2) * mt.cos(3 * x ** (3 / 2) + 2) - 81 / 4 * x * mt.sin(3 * x ** (3 / 2) + 2) + 2


class f3:
    def __call__(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return (x[0] - 3.3) ** 2 / 4 + (x[1] + 1.7) ** 2 / 15

    def grad(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List

        return np.array([(x[0] - 3.3) / 2, 2 * (x[1] + 1.7) / 15])

    def hess(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return np.array([[1 / 2, 0], [0, 2 / 15]])


class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return norm(x) ** 2

    def grad(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        b = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            b[i] = 2 * x[i]
        return b

    def hess(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        b = np.empty((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if i == j:
                    b[i][j] = 2
                else:
                    b[i][j] = 0
        return b


class Himmelblau:
    def __call__(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def grad(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        return np.array([4 * x[0] * ((x[0] ** 2) + x[1] - 11) + 2 * (x[0] + (x[1] ** 2) - 7),
                         2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + (x[1] ** 2) - 7)])

    def hess(self, x: np.ndarray):
        import numpy as np
        return np.array([[4 * (x[0] ** 2 + x[1] - 11) + 8 * x[0] ** 2 + 2, 4 * x[0] + 4 * x[1]],
                         [4 * x[0] + 4 * x[1], 4 * (x[1] ** 2 + x[0] - 7) + 8 * x[1] ** 2 + 2]])


class Rosenbrok:
    def __call__(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        b = 0
        n = len(x)
        for i in range(n - 1):
            b += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
        return b

    def grad(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        b = np.empty(len(x))
        n = len(x)
        b[0] = -400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0])
        if n == 2:
            b[1]=200*(x[n-1]-x[n-2]**2)
        elif n==3:
            b[1] = 200 * (x[1] - x[0] ** 2) - 400 * x[1] * (x[2] - x[1] ** 2) - 2 * (1 - x[1])
            b[2] = 200*(x[n-1]-x[n-2]**2)
        else:
            for i in range(1, n - 1):
                b[i] = 200*(x[i]-x[i-1]**2)-400*x[i]*(x[i+1]-x[i]**2)-2*(1-x[i])
            b[n-1] = 200*(x[n-1]-x[n-2]**2)
        return b
    def hess(self, x: np.ndarray):
        import numpy as np
        import scipy
        import math as mt
        from numpy.linalg import norm
        from typing import Callable, Tuple, Union, List
        b = np.empty([len(x), len(x)])
        n = len(x)
        if n == 2:
            b[0][0]=1200 * x[0] ** 2 + 2 - 400 * x[1]
            b[0][1]=-400*x[0]
            b[1][0]= b[0][1]
            b[1][1] = 200
        else:
            for i in range(len(x)-1):
                b[i][i]=202+1200*x[i]**2-400*x[i+1]
                b[i][i+1]=-400*x[i]
                for j in range(len(x)):
                    b[j][i]=b[i][j]
                b[0][0] = 1200 * x[0] ** 2 + 2 - 400 * x[1]
                b[n - 1][n - 1] = 200
        return b



def minimize(
        func: Callable,
        x_init: np.ndarray,
        learning_rate: Callable = lambda x: 0.1,
        method: str = 'gd',
        max_iter: int = 10_000,
        stopping_criteria: str = 'function',
        tolerance: float = 1e-2,
) -> Tuple:
    """
    Args:
        func: функция, у которой необходимо найти минимум (объект класса, который только что написали)
            (у него должны быть методы: __call__, grad, hess)
        x_init: начальная точка
        learning_rate: коэффициент перед направлением спуска
        method:
            "gd" - Градиентный спуск
            "newtone" - Метод Ньютона
        max_iter: максимально возможное число итераций для алгоритма
        stopping_criteria: когда останавливать алгоритм
            'points' - остановка по норме разности точек на соседних итерациях
            'function' - остановка по норме разности значений функции на соседних итерациях
            'gradient' - остановка по норме градиента функции
        tolerance: c какой точностью искать решение (участвует в критерии остановки)
    Returns:
        x_opt: найденная точка локального минимума
        points_history_list: (list) список с историей точек
        functions_history_list: (list) список с историей значений функции
        grad_history_list: (list) список с исторей значений градиентов функции
    """
    from numpy.linalg import norm
    assert max_iter > 0, 'max_iter должен быть > 0'
    assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
    assert stopping_criteria in ['points', 'function', 'gradient'], \
        'stopping_criteria can be "points", "function" or "gradient"!'

    points_history_list=[x_init]
    functions_history_list=[func(x_init)]
    grad_history_list=[func.grad(x_init)]
    i=0
    if method =="gd":
        if stopping_criteria == "points":
            while i < max_iter:
                points_history_list.append(points_history_list[i]-learning_rate(0.01)*grad_history_list[i])
                i+=1
                grad_history_list.append(func.grad(points_history_list[i]))
                functions_history_list.append(func(points_history_list[i]))
                x_opt=points_history_list[i]
                if norm(points_history_list[i]-points_history_list[i-1])<tolerance:
                    break
        elif stopping_criteria == "function":
            while i < max_iter:
                points_history_list.append(points_history_list[i]-learning_rate(0.01)*grad_history_list[i])
                i+=1
                grad_history_list.append(func.grad(points_history_list[i]))
                functions_history_list.append(func(points_history_list[i]))
                x_opt=points_history_list[i]
                if norm(functions_history_list[i]-functions_history_list[i-1])<tolerance:
                    break
        else:
            while i < max_iter:
                points_history_list.append(points_history_list[i] - learning_rate(0.01) * grad_history_list[i])
                i+=1
                grad_history_list.append(func.grad(points_history_list[i]))
                functions_history_list.append(func(points_history_list[i]))
                x_opt = points_history_list[i]
                if norm(grad_history_list[i]) < tolerance:
                    break
    elif method =="newtone":
        if isinstance(x_init, int) or isinstance(x_init, float):
            if stopping_criteria == "points":
                while i < max_iter:
                    points_history_list.append(points_history_list[i]-learning_rate(0.01)*grad_history_list[i]/func.hess(points_history_list[i]))
                    i+=1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt=points_history_list[i]
                    if norm(points_history_list[i]-points_history_list[i-1])<tolerance:
                        break
            elif stopping_criteria == "function":
                while i < max_iter:
                    points_history_list.append(points_history_list[i] - learning_rate(0.01) * grad_history_list[i]/func.hess(points_history_list[i]))
                    i += 1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt = points_history_list[i]
                    if norm(functions_history_list[i] - functions_history_list[i - 1]) < tolerance:
                        break
            else:
                while i < max_iter:
                    points_history_list.append(points_history_list[i] - learning_rate(0.01) * grad_history_list[i]/func.hess(points_history_list[i]))
                    i += 1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt = points_history_list[i]
                    if norm(grad_history_list[i]) < tolerance:
                        break
        else:
            if stopping_criteria == "points":
                while i < max_iter:
                    points_history_list.append(points_history_list[i] - learning_rate(0.01) * np.matmul(
                        np.linalg.inv(func.hess(points_history_list[i])), grad_history_list[i]))
                    i+=1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt=points_history_list[i]
                    if norm(points_history_list[i]-points_history_list[i-1])<tolerance:
                        break
            elif stopping_criteria == "function":
                while i < max_iter:
                    points_history_list.append(points_history_list[i] - learning_rate(0.01) * np.matmul(
                        np.linalg.inv(func.hess(points_history_list[i])), grad_history_list[i]))
                    i += 1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt = points_history_list[i]
                    if norm(functions_history_list[i] - functions_history_list[i - 1]) < tolerance:
                        break
            else:
                while i < max_iter:
                    points_history_list.append(points_history_list[i] - learning_rate(0.01) * np.matmul(
                        np.linalg.inv(func.hess(points_history_list[i])), grad_history_list[i]))
                    i += 1
                    grad_history_list.append(func.grad(points_history_list[i]))
                    functions_history_list.append(func(points_history_list[i]))
                    x_opt = points_history_list[i]
                    if norm(grad_history_list[i]) < tolerance:
                        break
    return x_opt, points_history_list, functions_history_list,grad_history_list

