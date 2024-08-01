from math import inf
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy import cumsum
import time

class empty_bee():
    Position=[]
    Cost=[]


class newbee():
    Position = []
    Cost = []

def RouletteWheelSelection(P):
    r = np.random.rand()
    C = cumsum(P)
    i = np.find(r <= C, 1, 'first')
    return i

def ABC(Position, VarMin, VarMax, CostFunction,  MaxIt):
    ## Problem Definition


    nVar = Position.shape[1]

    VarSize = np.array([1, nVar])



    nPop = Position.shape[0]

    nOnlooker = nPop

    L = np.round(0.6 * nVar * nPop)

    a = 1

    ## Initialization
    # Empty Bee Structure
    empty_bee.Position = []
    empty_bee.Cost = []
    # Initialize Population Array
    pop = np.matlib.repmat(empty_bee, nPop, 1)
    # Initialize Best Solution Ever Found
    #BestSol.Cost = inf
    # Create Initial Population
    for i in np.arange(1, nPop + 1).reshape(-1):
        pop[i].Position = np.uniform(VarMin, VarMax, VarSize)
        pop[i].Cost = CostFunction(pop[i].Position)
        if pop[i].Cost <= BestSol.Cost:
            BestSol = pop[i]

    # Abandonment Counter
    C = np.zeros((nPop, 1))
    # Array to Hold Best Cost Values
    BestCost = np.zeros((MaxIt, 1))
    ct = time.time()
    ## ABC Main Loop
    for it in np.arange(1, MaxIt + 1).reshape(-1):
        # Recruited Bees
        for i in np.arange(1, nPop + 1).reshape(-1):
            # Choose k randomly, not equal to i
            K = np.array([np.arange(1, i - 1 + 1), np.arange(i + 1, nPop + 1)])
            k = K(np.random.rand(np.array([1, np.asarray(K).size])))
            # Define Acceleration Coeff.
            phi = a * np.uniform(- 1, + 1, VarSize)
            # New Bee Position
            newbee.Position = pop[i].Position + np.multiply(phi, (pop[i].Position - pop[k].Position))
            # Evaluation
            newbee.Cost = CostFunction(newbee.Position)
            # Comparision
            if newbee.Cost <= pop[i].Cost:
                pop[i] = newbee
            else:
                C[i] = C[i] + 1
        # Calculate Fitness Values and Selection Probabilities
        F = np.zeros((nPop, 1))
        MeanCost = np.mean(np.array([pop.Cost]))
        for i in np.arange(1, nPop + 1).reshape(-1):
            F[i] = np.exp(- pop[i].Cost / MeanCost)
        P = F / sum(F)
        # Onlooker Bees
        for m in np.arange(1, nOnlooker + 1).reshape(-1):
            # Select Source Site
            i = RouletteWheelSelection(P)
            # Choose k randomly, not equal to i
            K = np.array([np.arange(1, i - 1 + 1), np.arange(i + 1, nPop + 1)])
            k = K(np.random.rand(np.array([1, np.asarray(K).size])))
            # Define Acceleration Coeff.
            phi = a * np.uniform(- 1, + 1, VarSize)
            # New Bee Position
            newbee.Position = pop[i].Position + np.multiply(phi, (pop[i].Position - pop[k].Position))
            # Evaluation
            newbee.Cost = CostFunction(newbee.Position)
            # Comparision
            if newbee.Cost <= pop[i].Cost:
                pop[i] = newbee
            else:
                C[i] = C[i] + 1
        # Scout Bees
        for i in np.arange(1, nPop + 1).reshape(-1):
            if C[i] >= L:
                pop[i].Position = np.uniform(VarMin, VarMax, VarSize)
                pop[i].Cost = CostFunction(pop[i].Position)
                C[i] = 0
        # Update Best Solution Ever Found
        for i in np.arange(1, nPop + 1).reshape(-1):
            if pop[i].Cost <= BestSol.Cost:
                BestSol = pop[i]
        # Store Best Cost Ever Found
        BestCost[it] = BestSol.Cost

    ct = time.time() - ct
    return BestSol.Cost, BestCost, BestSol, ct