import math
import numpy as np


def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def u_fun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def f1(x):
    # s = numpy.sum(x ** 2)
    s = 0
    for i in range(0, len(x)):
        s += x[i] ** 2
    return s


def f2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def f3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


def f4(x):
    o = max(abs(x))
    return o


def f5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0: dim - 1] ** 2)) ** 2 + (x[0: dim - 1] - 1) ** 2
    )
    return o


def f6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def f7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def f8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def f9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


def f10(x):
    dim = len(x)
    o = (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
            - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
            + 20
            + np.exp(1)
    )
    return o


def f11(x):
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


def f12(x):
    dim = len(x)
    o = (math.pi / dim) * (
            10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
            + np.sum((((x[: dim - 1] + 1) / 4) ** 2) * (1 + 10 * (np.sin(math.pi * (1 + (x[1:] + 1) / 4))) ** 2))
            + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(u_fun(x, 10, 100, 4))
    return o


def f13(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    o = 0.1 * (
            (np.sin(3 * np.pi * x[:, 0])) ** 2
            + np.sum((x[:, :-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[:, 1:])) ** 2), axis=1)
            + ((x[:, -1] - 1) ** 2) *
            (1 + (np.sin(2 * np.pi * x[:, -1])) ** 2)
    ) + np.sum(u_fun(x, 5, 100, 4))
    return o


def f14(x):
    a_s = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    a_s = np.asarray(a_s)
    b_s = np.zeros(25)
    v = np.matrix(x)
    for i in range(0, 25):
        h = v - a_s[:, i]
        b_s[i] = np.sum((np.power(h, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + np.sum(1.0 / (w + b_s))) ** (-1)
    return o


def f15(i):
    a_k = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    b_k = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    a_k = np.asarray(a_k)
    b_k = np.asarray(b_k)
    b_k = 1 / b_k
    fit = np.sum(
        (a_k - ((i[0] * (b_k ** 2 + i[1] * b_k)) / (b_k ** 2 + i[2] * b_k + i[3]))) ** 2
    )
    return fit


def f16(i):
    o = (
            4 * (i[0] ** 2)
            - 2.1 * (i[0] ** 4)
            + (i[0] ** 6) / 3
            + i[0] * i[1]
            - 4 * (i[1] ** 2)
            + 4 * (i[1] ** 4)
    )
    return o


def f17(i):
    o = (
            (i[1] - (i[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * i[0] - 6)
            ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(i[0])
            + 10
    )
    return o


def f18(i):
    o = (
                1
                + (i[0] + i[1] + 1) ** 2
                * (
                        19
                        - 14 * i[0]
                        + 3 * (i[0] ** 2)
                        - 14 * i[1]
                        + 6 * i[0] * i[1]
                        + 3 * i[1] ** 2
                )
        ) * (
                30
                + (2 * i[0] - 3 * i[1]) ** 2
                * (
                        18
                        - 32 * i[0]
                        + 12 * (i[0] ** 2)
                        + 48 * i[1]
                        - 36 * i[0] * i[1]
                        + 27 * (i[1] ** 2)
                )
        )
    return o


# map the inputs to the function blocks
def f19(j):
    a_h = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    a_h = np.asarray(a_h)
    c_h = [1, 1.2, 3, 3.2]
    c_h = np.asarray(c_h)
    p_h = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    p_h = np.asarray(p_h)
    o = 0
    for i in range(0, 4):
        o = o - c_h[i] * \
            np.exp(-(np.sum(a_h[i, :] * ((j - p_h[i, :]) ** 2))))
    return o


def f20(j):
    a_h = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    a_h = np.asarray(a_h)
    c_h = [1, 1.2, 3, 3.2]
    c_h = np.asarray(c_h)
    p_h = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    p_h = np.asarray(p_h)
    o = 0
    for i in range(0, 4):
        o = o - c_h[i] * \
            np.exp(-(np.sum(a_h[i, :] * ((j - p_h[i, :]) ** 2))))
    return o


def f21(j):
    a_sh = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    c_sh = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    a_sh = np.asarray(a_sh)
    c_sh = np.asarray(c_sh)
    fit = 0
    for i in range(5):
        v = np.matrix(j - a_sh[i, :])
        fit = fit - (v * v.T + c_sh[i]) ** (-1)
    o = fit.item(0)
    return o


def f22(j):
    a_sh = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    c_sh = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    a_sh = np.asarray(a_sh)
    c_sh = np.asarray(c_sh)
    fit = 0
    for i in range(7):
        v = np.matrix(j - a_sh[i, :])
        fit = fit - (v * v.T + c_sh[i]) ** (-1)
    o = fit.item(0)
    return o


def f23(j):
    a_sh = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    c_sh = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    a_sh = np.asarray(a_sh)
    c_sh = np.asarray(c_sh)
    fit = 0
    for i in range(10):
        v = np.matrix(j - a_sh[i, :])
        fit = fit - (v * v.T + c_sh[i]) ** (-1)
    o = fit.item(0)
    return o


def ea_som(variables_values):
    x1, x2 = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return func_value


def get_function_details(a):
    param = {
        "f1": ["f1", -100, 100, 30],
        "f2": ["f2", -10, 10, 30],
        "f3": ["f3", -100, 100, 30],
        "f4": ["f4", -100, 100, 30],
        "f5": ["f5", -30, 30, 30],
        "f6": ["f6", -100, 100, 30],
        "f7": ["f7", -1.28, 1.28, 30],
        "f8": ["f8", -500, 500, 30],
        "f9": ["f9", -5.12, 5.12, 30],
        "f10": ["f10", -32, 32, 30],
        "f11": ["f11", -600, 600, 30],
        "f12": ["f12", -50, 50, 30],
        "f13": ["f13", -50, 50, 30],
        "f14": ["f14", -65.536, 65.536, 2],
        "f15": ["f15", -5, 5, 4],
        "f16": ["f16", -5, 5, 2],
        "f17": ["f17", -5, 15, 2],
        "f18": ["f18", -2, 2, 2],
        "f19": ["f19", 0, 1, 3],
        "f20": ["f20", 0, 1, 6],
        "f21": ["f21", 0, 10, 4],
        "f22": ["f22", 0, 10, 4],
        "f23": ["f23", 0, 10, 4],
    }
    return param.get(a, "nothing")
