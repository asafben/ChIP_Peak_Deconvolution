# Imports #
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stts  # for stats.norm
from sklearn.metrics import mean_squared_error as mserror
import math  # for math.sqrt
from scipy.optimize import nnls, fmin
import time


# Constants #
GEN_REGION_BP_SIZE = 50
SIGMA = 3
GEN_POS = 4
READ_POS_INDEX = 2
DATA_PATH = r"synth_bind.tab"
N = 10
k = 5


# Helper Functions #
def RMSE(init_vec, Y, k):
    """
    We want to minimize this function.
    :param intit_vec: Vec of intial x and H.
    :param Y: in R^50
    :param k: Expected number of peaks in the data
    :return: Predicted RMSE error.
    """
    pred_y_val = predict(init_vec[:k], init_vec[k:])
    if np.isnan(pred_y_val).any():
        return np.inf
    return math.sqrt(mserror(Y, pred_y_val))


def plot_question_4(N, gen_pos, k):
    """
    As the name suggests.
    :return:
    """
    x, h, rmsd = opt_wrapper(N, gen_pos, k)
    plt.plot(np.arange(0, GEN_REGION_BP_SIZE), predict(x, h), label = "k = " + str(k))

    data = np.loadtxt(r"synth_bind.tab", delimiter='\t', usecols=range(4))
    Y = data[:, 3].T
    plt.plot(np.arange(0, GEN_REGION_BP_SIZE), Y, label="k = " + str(k))

    plt.show()


def plot_question_5():
    """
    As the name suggests.
    :return:
    """
    for cur_pos in range(GEN_POS):
        for cur_k in range(1, k + 1):
            x, h, rmse = opt_wrapper(N, cur_pos, cur_k)
            plt.plot(np.sum([ker(x[j])*h[j] for j in range(len(x))], axis=0), label='k: {:d}, RMSE: {:f}'.format(cur_k, rmse))
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

        data = np.loadtxt(r"synth_bind.tab", delimiter='\t', usecols=range(4))
        Y = data[:, cur_pos].T
        plt.plot(np.arange(0, GEN_REGION_BP_SIZE), Y, c='black', label='Actual Data')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


def RMSE2(X, Y, h):
    """
    We want to minimize this function.
    :param intit_vec: Vec of intial x and H.
    :param Y: in R^50
    :param k: Expected number of peaks in the data
    :return: Predicted RMSE error.
    """
    pred_y_val = predict(X, h)
    if np.isnan(pred_y_val).any():
        return np.inf
    return math.sqrt(mserror(Y, pred_y_val))


def plot_question_6(x, h, Y):
    """
    As the name suggests.
    :return:
    """
    plt.plot(np.arange(0, GEN_REGION_BP_SIZE), predict(x, h), label = "k = " + str(k))
    plt.plot(np.arange(GEN_REGION_BP_SIZE), Y, label="original")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


# Answers to Questions Functions #
# Question 1 #
def ker(x):
    """
    :param x: Peak center.
    :return: Vector of values of length 50 which represents the expected shape of the ChIP-seq for a single binding event.
    """

    return stts.norm.pdf(np.arange(GEN_REGION_BP_SIZE), x, SIGMA)


# Question 2 #
def predict(X, H):
    """
    Calculates the expected signal (a vector of length 50)
    :param X: [GEN_REGION_BP_SIZE] ^ n
    :param H: in R^n
    :return: If one of the values of H is negative return a vector of NaN values.
             Else returns the expected signal vector.
    """
    n = len(X)
    y = n * [np.zeros(GEN_REGION_BP_SIZE)]
    for i in range(n):
        if H[i] < 0:
            return np.full(GEN_REGION_BP_SIZE, np.nan)
        y[i] = ker(X[i]) * H[i]
    return np.sum(y, axis=0)


# Question 3 #
def optimize(Y, k):
    """
    Finds the vectors X, H that were most likely to generate the data.
    :param Y: in R^50
    :param k: expected number of peaks in the data
    :return: X, H, root mean squared error (RMSE) of the result.
    """
    init_val_X = np.array(Y.argsort()[GEN_REGION_BP_SIZE - k - 1: -1])
    init_val_H = np.array(np.sort(Y)[GEN_REGION_BP_SIZE - k - 1 : -1])
    init_vec = np.vstack((init_val_X, init_val_H))
    opt_X_H, min_RMSE, _, _, _ = fmin(RMSE, init_vec, args=(Y, k), full_output=True)
    return opt_X_H[:k], opt_X_H[k:], min_RMSE


# Question 4 #
def opt_wrapper(N, gen_pos, k):
    """
    Wrapper function that runs the optimization N times on one of the genomic positions (i.e. 1/2/3/4),
    records the RMSE value of each run and chooses the best set of parameters.
    :param N: optimization times
    :param gen_pos: out of 4
    :param k: expected number of peaks in the data
    :return: best set of parameters
    """
    opt_res = []
    data = np.loadtxt(DATA_PATH, delimiter='\t', usecols=range(4))
    selected_gen_pos = data[:, gen_pos].T

    for i in range(N):
        opt_res.append(optimize(selected_gen_pos, k))
    best_RMSE_index = np.argmin([i[2] for i in opt_res])
    return opt_res[best_RMSE_index]


# Question 5 #
def optimize_all_regions():
    """
    Execute the optimization on all four genomic position with k = 1 : 5, and N = 10. Provide figures that describe your
    results, and discuss them. Use as little figures as possible to convey your discussion points.
    :return:
    """
    plot_question_5()


# Question 6 #
def optimize2(Y, k):
    """
    Finds the vectors X, H that were most likely to generate the data.
    :param Y: in R^50
    :param k: expected number of peaks in the data
    :return: X, H, root mean squared error (RMSE) of the result.
    """
    init_val_X = np.array(Y.argsort()[GEN_REGION_BP_SIZE - k - 1: -1])
    A = np.zeros((GEN_REGION_BP_SIZE, k))

    for i in range(k):
        A[:, i] = ker(init_val_X[i])

    min_H = nnls(A, Y)[0]

    opt_X, min_RMSE, _, _, _ = fmin(RMSE2, init_val_X, args=(Y, min_H), full_output=True)
    return opt_X, min_H, min_RMSE


def compare_RMSE(Y, k):
    """
    Compare the running time and the RMSE of both versions: optimize and optimize2.
    :param Y:
    :param k:
    :return:
    """
    start1 = time.time()
    x, h, rmse1 = optimize(Y, k)
    linear_reg = time.time() - start1
    plot_question_6(x, h, Y)


    start2 = time.time()
    x, h, rmse2 = optimize2(Y, k)
    numerically = time.time() - start2
    plot_question_6(x, h, Y)

    print("Original Optimization: " + str(linear_reg) + ", With RMSE: " + str(rmse1))
    print("Modified Optimization: " + str(numerically) + ", With RMSE: " + str(rmse2))


# Main #
plot_question_4(N, 3, k)

plot_question_5()

data = np.loadtxt(r"synth_bind.tab", delimiter='\t', usecols=range(4))
Y = data[:, 3].T
compare_RMSE(Y, k)
