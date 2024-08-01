import numpy as np
import math

def evaluate_error(actual1,pred1):
    pred = pred1
    act = actual1
    r1 = pred
    x1 = act
    r1[np.where(r1 == 0)] = 1
    x1[np.where(x1 == 0)] = 1
    md = (100 / len(x1)) * sum(abs((r1 - x1) / r1))
    smape = (1 / len(x1)) * sum(abs((r1 - x1))/ ((abs(r1) + abs(x1))/ 2))
    points = np.zeros((len(x1), 1))
    for j in range (2,len(x1)):
        points[j] = abs(x1[j] - x1[j - 1])
    mase = sum(abs((r1 - x1))) / ((len(x1) / (len(x1) - 1)) * sum(points))
    mae = np.mean(abs(r1 - x1))
    rmse = math.sqrt(np.mean(abs(x1 - r1)** 2))
    onenorm = sum(abs(r1 - x1))
    twonorm = math.sqrt(sum((abs(r1 - x1))** 2))
    infinitynorm = max(abs(r1 - x1))
    val1 = [md, smape, mase, mae, rmse, onenorm, twonorm, infinitynorm]
    vals = np.asarray(val1)
    return vals


