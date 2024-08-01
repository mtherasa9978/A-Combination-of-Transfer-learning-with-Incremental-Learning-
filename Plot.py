import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def plot_results():
    Error = np.load('err_meas.npy', allow_pickle=True)
    Terms = ['MEP','SMAPE','MASE','MAE','RMSE','L1_Norm', 'L2_Norm' ,'L_InfinityNorm']
    Algorithm = ['TERMS', 'SFO', 'WOA', 'SFO_WOA', 'DHOA','PROPOSED(P1)','PROPOSED(P2)']
    Classifier = ['TERMS', 'RNN', 'NN', 'DNN','PROPOSED(P1)','PROPOSED(P2)']
    lnn = ['Remember', 'PerfectRecall', 'MeanGoodness']

    for d in range(3):
        if d == 0:
            Error_eval = np.load('Error_eval1.npy')
        elif d == 1:
            Error_eval = np.load('Error_eval2.npy')
        else:
            Error_eval = np.load('Error_eval3.npy')

        Eval_Table1 = Error[d, 0:5, :]
        Table1 = PrettyTable()
        Table1.add_column(Algorithm[0], Terms)
        for j in range(0, len(Algorithm)-1):
            if j == 4:
                p1 = Error_eval[d, 4, :]
                Table1.add_column(Algorithm[j+1], p1)
            elif j == 5:
                Table1.add_column(Algorithm[j + 1], Eval_Table1[j-1])
            else:
                Table1.add_column(Algorithm[j + 1], Eval_Table1[j])
        print('---------------------------------------- Data ', str(d + 1),
              ' Algorithm Comparison ----------------------------------------')
        print(Table1)

        Eval_Table2 = Error[d, 5:8, :]
        Table2 = PrettyTable()
        Table2.add_column(Classifier[0], Terms)
        for j in range(0, len(Classifier) - 1):
            if j == 3:
                pp1 = Error_eval[d, 4, :]
                Table2.add_column(Classifier[j + 1], pp1)
            elif j == 4:
                Table2.add_column(Classifier[j + 1], Eval_Table1[4])
            else:
                Table2.add_column(Classifier[j + 1], Eval_Table2[j])
        print('---------------------------------------- Data ', str(d + 1),
              ' Classifier Comparison ----------------------------------------')
        print(Table2)


    for i in range(3): # rem,perf,mean goodness
        if i == 0:
            data1 = np.load('Rem_p2.npy')
            data2 = np.load('Remem.npy')
        elif i == 1:
            data1 = np.load('Percall_p2.npy')
            data2 = np.load('Perf_recall.npy')
        elif i == 2:
            data1 = np.load('Gns_p2.npy')
            data2 =np.load('Mean_Gd.npy')

        val = np.zeros((6, 3))
        for j in range(6):   # for all algorithms
            for k in range(3):  # for all test cases
                if j == 4:
                    val[j, k] = data2[4,k]
                elif j == 5:
                    val[j, k] = data1[k,4]
                else:
                    val[j, k] = data1[k,j]

        value1 = np.zeros((4, 3))
        value1[0:2, :] = val[0:2, :]
        value1[2, :] = val[3, :]
        value1[3, :] = val[5, :]
        n_groups = 3
        data = value1
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.10
        opacity = 1
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='b',
                label='SFO-DNN [27]')
        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='g',
                label='WOA-DNN [28]')
        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='y',
                label='DHOA-DNN [26]')
        plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='m',
                label='M-DHOA-DNN')
        plt.ylabel(lnn[i])
        plt.xticks(index + bar_width,
                   ('Testcase1','Testcase2','Testcase3'))
        plt.legend(loc=4)
        plt.tight_layout()
        path1 = "./Results/perfAlg-%s.png" % (i)
        plt.savefig(path1)
        plt.show()

    for i in range(3):  # rem,perf,mean goodness
        if i == 0:
            data1 = np.load('Rem_p2.npy')
            data2 = np.load('Remem.npy')
        elif i == 1:
            data1 = np.load('Percall_p2.npy')
            data2 = np.load('Perf_recall.npy')
        elif i == 2:
            data1 = np.load('Gns_p2.npy')
            data2 =np.load('Mean_Gd.npy')

        value = np.zeros((5, 3))
        for jj in range(5):  # for all methods
            for kk in range(3):  # for all test cases
                if jj == 3:
                    value[jj, kk] = data2[4, kk]
                elif jj == 4:
                    value[jj, kk] = data1[kk, 4]
                else:
                    value[jj, kk] = data1[kk, 5+jj]

        value2 = np.zeros((4, 3))
        value2[0:3, :] = value[0:3, :]
        value2[3, :] = value[4, :]
        n_groups = 3
        data = value2
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.10
        opacity = 1
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='b',
                label='RNN [29]')
        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='g',
                label='NN [30]')
        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='y',
                label='DNN [31]')
        plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='m',
                label='M-DHOA-DNN')
        plt.ylabel(lnn[i])
        plt.xticks(index + bar_width,
                   ('Testcase1', 'Testcase2', 'Testcase3'))
        plt.legend(loc=4)
        plt.tight_layout()
        path1 = "./Results/perfCls-%s.png" % (i)
        plt.savefig(path1)
        plt.show()


