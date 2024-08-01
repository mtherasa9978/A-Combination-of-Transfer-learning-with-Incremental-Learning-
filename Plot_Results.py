import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def Plot_Results():
    Error3 = np.load('err_meas_f.npy', allow_pickle=True)
    Error2 = np.load('err_meas.npy', allow_pickle=True)
    Terms = ['MEP','SMAPE','MASE','MAE','RMSE','L1_Norm', 'L2_Norm' ,'L_InfinityNorm']
    Algorithm = ['TERMS', 'SFO-WOA', 'IMP-DHOA', 'AOX', 'ABC', 'PROPOSED(P3)']
    Classifier = ['TERMS', 'NN', 'DNN', 'RNN','LSTM', 'PROPOSED(P1)','PROPOSED(P2)', 'PROPOSED(P3)']
    lnn = ['Remember', 'PerfectRecall', 'MeanGoodness']

    for d in range(3):
        if d == 0:
            Error_eval = np.load('Error_eval1.npy')
        elif d == 1:
            Error_eval = np.load('Error_eval2.npy')
        else:
            Error_eval = np.load('Error_eval3.npy')

        Eval_Table2 = Error2[d, :, :]
        Eval_Table3 = Error3[d, :, :]
        Table1 = PrettyTable()
        Table1.add_column(Algorithm[0], Terms)
        for j in range(0, len(Algorithm)-1):
            # if j == 4:
            #     p1 = Error_eval[d, 4, :]
            #     Table1.add_column(Algorithm[j+1], p1)
            # elif j == 5:
            #     Table1.add_column(Algorithm[j + 1], Eval_Table2[j-1])
            # if j == 6:
            #     Table1.add_column(Algorithm[j + 1], Eval_Table3[j-2])
            # else:
            Table1.add_column(Algorithm[j + 1], Eval_Table3[j])
        print('---------------------------------------- Data ', str(d + 1),
              ' Algorithm Comparison ----------------------------------------')
        print(Table1)


        Table2 = PrettyTable()
        Table2.add_column(Classifier[0], Terms)
        for j in range(0, len(Classifier) - 1):
            if j == 4:
                pp1 = Error_eval[d, 4, :]
                Table2.add_column(Classifier[j + 1], pp1)
            elif j == 5:
                Table2.add_column(Classifier[j + 1], Eval_Table2[4])
            elif j == 6:
                Table2.add_column(Classifier[j + 1], Eval_Table3[4])
            else:
                Table2.add_column(Classifier[j + 1], Eval_Table3[j])
        print('---------------------------------------- Data ', str(d + 1),
              ' Classifier Comparison ----------------------------------------')
        print(Table2)

    Paper3 = np.load('meas_all.npy', allow_pickle=True)
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

        val = np.zeros((7, 3))
        for j in range(7):   # for all algorithms
            for k in range(3):  # for all test cases
                if j == 4:
                    val[j, k] = data2[4,k]
                elif j == 5:
                    val[j, k] = data1[k,4]
                elif j == 6:
                    val[j, k] = Paper3[i, 4, k]
                else:
                    val[j, k] = Paper3[i, j, k]
        data = val
        n_groups = 3


        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.10
        opacity = 1
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='#c0a117',
                label='SF - WOA - TLIQAS [28]')
        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='k',
                label='IMP - DHOA - TLIQAS[29]')
        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='g',
                label='AOX - TLIQAS [26]')
        plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='r',
                label='ABC - TLIQAS [27]')
        # plt.bar(index + 4 * bar_width, data[4, :], bar_width,
        #         alpha=opacity,
        #         color='y',
        #         label='SF - WOA - RNN [30]')
        # plt.bar(index + 5 * bar_width, data[5, :], bar_width,
        #         alpha=opacity,
        #         color='m',
        #         label='M - DHOA - DNN [31]')
        plt.bar(index + 4 * bar_width, data[6, :], bar_width,
                alpha=opacity,
                color='c',
                label='HA - ABCO - TLIQAS')
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

        value = np.zeros((7, 3))
        for jj in range(7):  # for all methods
            for kk in range(3):  # for all test cases
                if jj == 4:
                    value[jj, kk] = data2[4, kk]
                elif jj == 5:
                    value[jj, kk] = data1[kk, 4]
                elif jj == 6:
                    value[jj, kk] = Paper3[i, 4, kk]
                else:
                    value[jj, kk] = Paper3[i, jj+5, kk]

        n_groups = 3
        data = value
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.10
        opacity = 1
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='r',
                label='NN [32]')
        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='g',
                label='DNN [33]')
        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='b',
                label='RNN [34]')
        plt.bar(index + 3 * bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='m',
                label='LSTM [35]')
        plt.bar(index + 4 * bar_width, data[4, :], bar_width,
                alpha=opacity,
                color='k',
                label='SF - WOA - RNN [30]')
        plt.bar(index + 5 * bar_width, data[5, :], bar_width,
                alpha=opacity,
                color='#c0a117',
                label='M - DHOA - DNN [31]')
        plt.bar(index + 6 * bar_width, data[6, :], bar_width,
                alpha=opacity,
                color='c',
                label='HA - ABCO - TLIQAS')
        plt.ylabel(lnn[i])
        plt.xticks(index + bar_width,
                   ('Testcase1', 'Testcase2', 'Testcase3'))
        plt.legend(loc=4)
        plt.tight_layout()
        path1 = "./Results/perfCls-%s.png" % (i)
        plt.savefig(path1)
        plt.show()


Plot_Results()