
import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    """
    Return the joint CDF F_XY(x, y) for (X, Y) uniform on the unit square.

    F_XY(x, y) =
        0                   if x <= 0 or y <= 0
        x*y                 if 0 < x < 1 and 0 < y < 1
        x                   if 0 < x < 1 and y >= 1
        y                   if x >= 1 and 0 < y < 1
        1                   if x >= 1 and y >= 1
    """
    # return min(1,max(0, x)) * (min(1, max(0, y))
    if x <= 0 or y <= 0:
        return 0.0
    if x >= 1 and y >= 1:
        return 1.0
    if x >= 1:
        return y
    if y >= 1:
        return x
    return x * y


def rectangle_probability(x1, x2, y1, y2):
    """
    Compute P(x1 < X <= x2, y1 < Y <= y2)
    using the joint CDF rectangle formula.
    """
    all2 = joint_cdf_unit_square(x2, y2)
    x1y2 = joint_cdf_unit_square(x1, y2)
    x2y1 = joint_cdf_unit_square(x2, y1)
    all1 = joint_cdf_unit_square(x1, y1)
    return all2 - x1y2 - x2y1 + all1


def marginal_fx_unit_square(x):
    """
    Return the marginal PDF f_X(x) for X when (X, Y) is uniform on the unit square.

    f_X(x) =
        1   if 0 < x < 1
        0   otherwise
    """
    return float(0 < x < 1)

def marginal_fy_unit_square(y):
    """
    Return the marginal PDF f_Y(y) for Y when (X, Y) is uniform on the unit square.

    f_Y(y) =
        1   if 0 < y < 1
        0   otherwise
    """
    return float(0 < y < 1)


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):
    """
    Return P_XY(x, y) for:
    X = number of heads in the first toss
    Y = total number of heads in both tosses

    Table:
                 y=0   y=1   y=2
        x=0      1/4   1/4    0
        x=1       0    1/4   1/4
    """
    hash_table = {(0,0), (0,1), (1,1), (1,2)}
    return 0.25 if (x,y) in hash_table else 0.0

def marginal_px_heads(x):
    """
    Return P_X(x) by summing the joint PMF over y.
    """
    return 0.5 if x in (0,1) else 0


def marginal_py_heads(y):
    """
    Return P_Y(y) by summing the joint PMF over x.
    """
    if y == 0:
        return 0.25
    if y == 1:
        return 0.5
    if y == 2:
        return 0.25
    return 0.0


def check_independence_heads():
    """
    Return True if X and Y are independent, else False.
    """
    PX = {0: 0.5, 1: 0.5}
    PY = {0: 0.25, 1: 0.5, 2: 0.25}

    PXY = {
        (0,0): 0.25, (0,1): 0.25,
        (1,1): 0.25, (1,2): 0.25
    }

    for (x, y), p in PXY.items():
        if abs(p - PX[x] * PY[y]) > 1e-12:
            return False
    return True
