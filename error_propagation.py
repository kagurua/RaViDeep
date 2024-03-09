import sympy
import math
import json
import numpy as np
import dill
dill.settings['recurse'] = True
with open('./data/calib.txt', 'r') as f:
    calib = json.load(f)
Intrinsic = calib['Intrinsic']


def calculate_function(filename2save):
    # model xyz
    r = sympy.Symbol('r')
    t = sympy.Symbol('t')
    p = sympy.Symbol('p')

    # set original sigma
    sigma_2_r = 0.1
    sigma_2_t = (math.radians(1.4) / sympy.cos(t)) ** 2
    sigma_2_p = (math.radians(18) / sympy.cos(p)) ** 2

    # function to calculate
    x = r * sympy.cos(p) * sympy.sin(t)
    y = r * sympy.cos(p) * sympy.cos(t)
    z = r * sympy.sin(p)

    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    g = sympy.Symbol('g')
    t1 = sympy.Symbol('t1')
    t2 = sympy.Symbol('t2')
    t3 = sympy.Symbol('t3')

    # set original sigma
    sigma_2_a = 0.001 ** 2  # alpha
    sigma_2_b = 0.001 ** 2  # beta
    sigma_2_g = 0.001 ** 2  # gamma
    sigma_2_t1 = 0.01 ** 2  # tx
    sigma_2_t2 = 0.01 ** 2  # ty
    sigma_2_t3 = 0.01 ** 2  # tz

    # calculate translation
    A = sympy.Matrix(Intrinsic)
    Rx = sympy.Matrix([[1, 0, 0], [0, sympy.cos(a), sympy.sin(a)], [0, -sympy.sin(a), sympy.cos(a)]])
    Ry = sympy.Matrix([[sympy.cos(b), 0, -sympy.sin(b)], [0, 1, 0], [sympy.sin(b), 0, sympy.cos(b)]])
    Rz = sympy.Matrix([[sympy.cos(g), -sympy.sin(g), 0], [sympy.sin(g), sympy.cos(g), 0], [0, 0, 1]])
    R = Rx * Ry * Rz
    T = sympy.Matrix([t1, t2, t3])
    B = R.col_insert(-1, T)
    H = A * B
    # calculate projection
    udvdd = H * sympy.Matrix([[x], [y], [z], [1]])
    uv1 = udvdd / udvdd[2, 0]

    # function
    u = uv1[0, 0]
    v = uv1[1, 0]
    d = udvdd[2, 0]

    sigma_2_u = sympy.diff(u, r) * sympy.diff(u, r) * sigma_2_r \
                + sympy.diff(u, t) * sympy.diff(u, t) * sigma_2_t \
                + sympy.diff(u, p) * sympy.diff(u, p) * sigma_2_p \
                + sympy.diff(u, a) * sympy.diff(u, a) * sigma_2_a \
                + sympy.diff(u, b) * sympy.diff(u, b) * sigma_2_b \
                + sympy.diff(u, g) * sympy.diff(u, g) * sigma_2_g \
                + sympy.diff(u, t1) * sympy.diff(u, t1) * sigma_2_t1 \
                + sympy.diff(u, t2) * sympy.diff(u, t2) * sigma_2_t2 \
                + sympy.diff(u, t3) * sympy.diff(u, t3) * sigma_2_t3
    sigma_2_v = sympy.diff(v, r) * sympy.diff(v, r) * sigma_2_r \
                + sympy.diff(v, t) * sympy.diff(v, t) * sigma_2_t \
                + sympy.diff(v, p) * sympy.diff(v, p) * sigma_2_p \
                + sympy.diff(v, a) * sympy.diff(v, a) * sigma_2_a \
                + sympy.diff(v, b) * sympy.diff(v, b) * sigma_2_b \
                + sympy.diff(v, g) * sympy.diff(v, g) * sigma_2_g \
                + sympy.diff(v, t1) * sympy.diff(v, t1) * sigma_2_t1 \
                + sympy.diff(v, t2) * sympy.diff(v, t2) * sigma_2_t2 \
                + sympy.diff(v, t3) * sympy.diff(v, t3) * sigma_2_t3
    sigma_2_d = sympy.diff(d, r) * sympy.diff(d, r) * sigma_2_r \
                + sympy.diff(d, t) * sympy.diff(d, t) * sigma_2_t \
                + sympy.diff(d, p) * sympy.diff(d, p) * sigma_2_p \
                + sympy.diff(d, a) * sympy.diff(d, a) * sigma_2_a \
                + sympy.diff(d, b) * sympy.diff(d, b) * sigma_2_b \
                + sympy.diff(d, g) * sympy.diff(d, g) * sigma_2_g \
                + sympy.diff(d, t1) * sympy.diff(d, t1) * sigma_2_t1 \
                + sympy.diff(d, t2) * sympy.diff(d, t2) * sigma_2_t2 \
                + sympy.diff(d, t3) * sympy.diff(d, t3) * sigma_2_t3

    func_sigma_u = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_u), 'numpy')
    func_sigma_v = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_v), 'numpy')
    func_sigma_d = sympy.lambdify((r, t, p), sympy.sqrt(sigma_2_d), 'numpy')
    dill.dump((func_sigma_u, func_sigma_v, func_sigma_d), open(filename2save, 'wb'))


if __name__ == "__main__":
    saved_function_file = './data/error_function_saved.bin'
    calculate_function(saved_function_file)
    f_sigma_u, f_sigma_v, f_sigma_d = dill.load(open(saved_function_file, 'rb'))

    sigma_u_array = f_sigma_u(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    sigma_v_array = f_sigma_v(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))
    sigma_d_array = f_sigma_d(np.array([10, 20]), np.array([0, 20 / 180]), np.array([0, 45 / 180]))

    print(sigma_u_array)
    print(sigma_v_array)
    print(sigma_d_array)
