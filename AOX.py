from math import inf

import numpy as np
import scipy.special
import time


def AOX(X, LB, UB, F_obj, T):
    N, Dim = X.shape[0], X.shape[1]
    Best_P = np.zeros((1, Dim))
    Best_FF = inf

    Xnew = X
    Ffun = np.zeros((1, X.shape[1 - 1]))
    Ffun_new = np.zeros((1, Xnew.shape[1 - 1]))
    t = 1
    conv = np.zeros((T))
    alpha = 0.1
    delta = 0.1
    ct = time.time()
    while t < T + 1:

        for i in np.arange(1, X.shape[1 - 1] + 1).reshape(-1):
            F_UB = X[i, :] > UB
            F_LB = X[i, :] < LB
            X[i, :] = (np.multiply(X[i, :], (not (F_UB + F_LB)))) + np.multiply(UB, F_UB) + np.multiply(LB, F_LB)
            Ffun[1, i] = F_obj(X[i, :])
            if Ffun[1, i] < Best_FF:
                Best_FF = Ffun[1, i]
                Best_P = X[i, :]
        G2 = 2 * np.random.rand() - 1
        G1 = 2 * (1 - (t / T))
        to = np.arange(1, Dim + 1)
        u = 0.0265
        r0 = 10
        r = r0 + u * to
        omega = 0.005
        phi0 = 3 * np.pi / 2
        phi = - omega * to + phi0
        x = np.multiply(r, np.sin(phi))
        y = np.multiply(r, np.cos(phi))
        QF = t ** ((2 * np.random.rand() - 1) / (1 - T) ** 2)
        # -------------------------------------------------------------------------------------
        for i in np.arange(1, X.shape[1 - 1] + 1).reshape(-1):
            # -------------------------------------------------------------------------------------
            if t <= (2 / 3) * T:
                if np.random.rand() < 0.5:
                    Xnew[i, :] = Best_P[1, :] * (1 - t / T) + (np.mean(X[i, :]) - Best_P[1, :]) * np.random.rand()
                    Ffun_new[1, i] = F_obj(Xnew[i, :])
                    if Ffun_new[1, i] < Ffun[1, i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[1, i] = Ffun_new[1, i]
                else:
                    # -------------------------------------------------------------------------------------
                    Xnew[i, :] = np.multiply(Best_P[1, :], Levy(Dim)) + X[(int(np.floor(N * np.random.rand() + 1))),
                                                                        :] + (y - x) * np.random.rand()
                    Ffun_new[1, i] = F_obj(Xnew[i, :])
                    if Ffun_new[1, i] < Ffun[1, i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[1, i] = Ffun_new[1, i]
                # -------------------------------------------------------------------------------------
            else:
                if np.random.rand() < 0.5:
                    Xnew[i, :] = (Best_P[1, :] - np.mean(X)) * alpha - np.random.rand() + (
                                (UB - LB) * np.random.rand() + LB) * delta
                    Ffun_new[1, i] = F_obj(Xnew[i, :])
                    if Ffun_new[1, i] < Ffun[1, i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[1, i] = Ffun_new[1, i]

                else:
                    # -------------------------------------------------------------------------------------
                    Xnew[i, :] = QF * Best_P[1, :] - (G2 * X[i, :] * np.random.rand()) - np.multiply(G1, Levy(
                        Dim)) + np.random.rand() * G2
                    Ffun_new[1, i] = F_obj(Xnew[i, :])
                    if Ffun_new[1, i] < Ffun[1, i]:
                        X[i, :] = Xnew[i, :]
                        Ffun[1, i] = Ffun_new[1, i]

        conv[t] = Best_FF
        t = t + 1

    ct = time.time() - ct
    return Best_FF, Best_P, conv, ct


def Levy(d):
    beta = 1.5
    sigma = (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / np.abs(v) ** (1 / beta)
    o = step
    return o

