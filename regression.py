from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def bestfit(arr, params):
    """
    Args:
        arr [list]
        params [list]: List of coefficients with the lowest order coefficient
                       first. E.g., [c, b, a] for ax^2 + bx + c
    """
    return [sum([e ** i * p for i, p in enumerate(params)]) for e in arr]


def rmse(x_vec, y_vec, params):
    """UNUSED - instead, rely on statsmodel implementation to optimize"""
    def sq_err(x_pt, y_pt, params):
        x_pt = [x_pt] if type(x_pt) is not list else x_pt
        return (y_pt - sum([x * b for x, b in zip(x_pt + [1], params)])) ** 2
    return sum([sq_err(x_vec[i], y_vec[i], params) for i in range(len(x_vec))])


def regression_helper_2_var(x_mat, y_vec=None, row_major=True, x_labels=None,
                            y_label=None, plot=False):
    """
    Args:
        x_mat [list]: NxM matrix (N dimensions/rows, M samples per row). If
                      y_vec is supplied, solve in the format y_vec = B * x_mat.
                      Else, solve in the format x_mat[-1] = B * x_mat[:-1].
        y_vec [list]: (Optional) Mx1 vector.
        row_major [bool]: (Optional) If true, data is in the format:
                          [x0=(x00, x01, ..., x0M), ...].
                          Otherwise, data is in the format:
                          [dim_0=(x00, x10, ..., xN0), dim_1=(...), ...].
        x_labels [list]: Feature labels
        y_label [str]: Result label
    """
    # Transform x_mat if in row major format
    if row_major:
        x_mat = [[x[i] for x in x_mat] for i in range(len(x_mat[0]))]
    # Allow x_mat to be a vector
    if type(x_mat[0]) is not list:
        x_mat = [x_mat]
    X, y = (x_mat, y_vec) if y_vec else (x_mat[:-1], x_mat[-1])

    # Initialize other parameters
    colors = [0.5 for _ in range(len(y))]
    c2 = [0.5 for _ in range(len(y))]
    sizes = [np.pi * 5 ** 2 for _ in range(len(y))]  # Circle radii

    X_CONT_PREC = 1000
    for i, x_vec in enumerate(X):
        data = zip(x_vec, y, colors, sizes)
        # Remove null values and convert x, y pairs to floats
        data = [(float(tup[0]), float(tup[1]), tup[2], tup[3]) for tup in data
                if tup[0] is not None and tup[1] is not None]
        # Order by x value
        data = sorted(data, key=lambda x: x[0])
        x_sorted, y_sorted, c_sorted, s_sorted = (
            [d[0] for d in data],
            [d[1] for d in data],
            [d[2] for d in data],
            [d[3] for d in data]
        )

        x_cont = [float(n) * (max(x_sorted) - min(x_sorted)) / X_CONT_PREC + min(x_sorted)
                  for n in range(X_CONT_PREC)]  # Points from x_min to x_max
        model = sm.OLS(y_sorted, [(1, x) for x in x_sorted])  # Add intercept
        results = model.fit()
        if x_labels and y_label:
            print '%s vs %s R^2: %f' % (x_labels[i], y_label, results.rsquared)
        if plot:
            print results.summary()
            plt.scatter(x_sorted, y_sorted, c=c_sorted, s=s_sorted, alpha=0.5)
            lfit, = plt.plot(x_cont, bestfit(x_cont, results.params))
            plt.legend([lfit], ['Best Fit'])
            plt.show()

