from AntLionOptimizer import *
from Benchmarks import *
import Benchmarks
import inspect
from time import time
from pyMetaheuristic.utils import graphs


def open_graphs():
    plot_parameters = {
        'min_values': (-5, -5),
        'max_values': (5, 5),
        'step': (0.1, 0.1),
        'solution': [],
        'proj_view': '3D',
        'view': 'notebook'
    }
    graphs.plot_single_function(target_function=f1, **plot_parameters)


def test():
    functions = sorted(list(
        filter(lambda x: x[0].startswith("f"), inspect.getmembers(
            Benchmarks, inspect.isfunction))
    ), key=lambda x: int(x[0][1:]))

    for func in functions:
        func = func[1]
        function_name, lb, up, dim = Benchmarks.get_function_details(func.__name__)
        start_time = time()
        optimizer = AntLionOptimizer(n=30, dimension=dim, c=lb, d=up, fitness_function=func, max_iter=1000)
        best_solution, best_score = optimizer.optimize()
        time_s = time() - start_time
        print(f'function_name = {function_name}\n'
              f'best_score = {best_score}\n'
              f'time_s = {time_s}\n'
              f'best_solution = {best_solution}')
        print("__________________________________________________________________________"
              "______________________________________")


def main():
    # open_graphs()
    test()


if __name__ == "__main__":
    main()
