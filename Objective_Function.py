from Global_vars import Global_vars
import numpy as np
from Model_LSTM import Modified_Model_LSTM

def Objective_Function(Soln):

    Stucture = Global_vars.Stucture
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i])
            Selected_Feat = Global_vars.Feat[:, sol[:5].astype('int')]
            per = round(len(Selected_Feat) * (0.75))  # % of learning
            Train_Data = Selected_Feat[:per, :]
            Train_Target = Global_vars.Target[:per]
            Test_Data = Selected_Feat[per:, :]
            Test_Target = Global_vars.Target[per:]
            out = Modified_Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target,sol[5:], Stucture)
            act = Test_Target.reshape(len(Test_Target), 1)
            pred = out
            err = np.mean(abs(act - pred))
            Fitn[i] = err
        return Fitn

    else:
        sol = np.round(Soln)
        Selected_Feat = Global_vars.Feat[:, sol[:5].astype('int')]
        per = round(len(Selected_Feat) * (0.75))  # % of learning
        Train_Data = Selected_Feat[:per, :]
        Train_Target = Global_vars.Target[:per]
        Test_Data = Selected_Feat[per:, :]
        Test_Target = Global_vars.Target[per:]
        out = Modified_Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target,sol[5:-1], Stucture)
        act = Test_Target.reshape(len(Test_Target), 1)
        pred = out
        err = np.mean(abs(act - pred))
        Fitn = err
        return Fitn