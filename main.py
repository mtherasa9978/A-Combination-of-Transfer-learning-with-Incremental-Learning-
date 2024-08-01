import math
import os
import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
from bert_serving.client import BertClient
from sklearn.decomposition import PCA
from Model_LSTM import train_lstm
from random import random, uniform
from Objective_Function import Objective_Function
from numpy import matlib
from Global_vars import Global_vars
from SFO_WOA import SFO_WOA
from IMP_DHOA import IMP_DHOA
from AOX import AOX
from ABC import ABC
from PROPOSED import PROPOSED
from Model_LSTM import Modified__Model_LSTM, Model_LSTM
from Model_NN import Model_NN
from Model_DNN import Model_DNN
from Model_RNN import Model_RNN
from Plot_Results import Plot_Results


# Euclidean distance
def euclidean_distance(x, y):
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


# Removing puctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    return no_punct


## Read Dataset
an = 0
if an == 1:
    AllData = []
    for file1 in os.listdir("./Dataset"):  # For all dataset
        for file2 in os.listdir("./Dataset" + "/" + file1):
            file3 = file2 + '.xlsx'
            path1 = "./Dataset/" + file1 + "/" + file2 + "/%s" % file3
            wb = pd.ExcelFile(path1, engine='openpyxl')  # Read Excel file
            df1 = wb.parse('Sheet1')
            data = df1.values
            AllData.append(data)
        np.save('AllData.npy', AllData)

## Qn qnd Ans in strings
an = 0
if an == 1:
    Qn = []
    Ans = []
    Input = np.load('AllData.npy', allow_pickle=True)
    for i in range(len(Input)):  # for all data
        M = Input[i]
        Question = M[:, 1]
        Answer = M[:, 5]
        Qn.append(Question)
        Ans.append(Answer)
        np.save('Qn.npy', Qn)
        np.save('Ans.npy', Ans)

# Document Extraction
an = 0
if an == 1:
    Data = []
    a = ['Question_Answer_Dataset_v1.0', 'Question_Answer_Dataset_v1.1', 'Question_Answer_Dataset_v1.2']  # datasets
    b = ['S08', 'S09', 'S08', 'S09', 'S10', 'S08', 'S09', 'S10']  # datas in datasets
    Input = np.load('AllData.npy', allow_pickle=True)
    for i in range(0,Input.shape[0]):
        uniqueValues = np.unique(Input[i][:, 0])
        Data1 = []
        for j in range(uniqueValues.shape[0]):
            print(i, j)
            if i == 0 or i == 1:
                d = a[0]
            elif i == 2 or i == 3 or i == 4:
                d = a[1]
            else:
                d = a[2]
            index = Input[i][:, 0].tolist().index(uniqueValues[j])
            if type(Input[i][index, 2]) == float:
                data = 0
            else:
                l = Input[i][index, 2] + '.txt'
                c = str(b[i])
                e = "./Dataset/"
                one = "%s%s/%s/%s" % (e, d, c, l)
                s = l[5:]
                print(s)
                if ((i == 0 or i == 2 or i == 3 or i == 5 or i == 6) and s == 'set4/a8.txt') or (
                        (i == 1 or i == 3 or i == 6) and s == 'set3/a2.txt') or (
                        (i == 1 or i == 3 or i == 6) and s == 'set3/a6.txt') or (i== 1 and j== 1):
                    encode = 'cp1252'
                else:
                    encode = 'UTF-8'
                with open(one, encoding=encode) as file:
                    data = file.read()
            Data1.append(data)
        Data.append(Data1)
    np.save('Data.npy', Data)

# Feature Extraction using BERT and GLOVE -   for context
an = 0
if an == 1:
    bc = BertClient(ip="SERVER_IP_HERE")
    glove_filename = 'glove.6B.50d.txt'
    # glove_path = os.path.abspath(os.path.join('GLOVE', glove_filename))
    word2vec_output_file = glove_filename + '.word2vec'
    # glove2word2vec(glove_path, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    data = np.load('Data.npy', allow_pickle=True)
    Feat_Doc = []
    for i in range(5, 8):  # for all 3 dataset
        Data1 = []
        for j in range(len(data[i])):
            print(i, j, len(data))
            sin_data = data[i][j]
            if type(sin_data) == int:
                vector = 0
            else:
                li = sin_data.splitlines()
                li = list(filter(None, li))
                glove_vector = model.get_vector(li[0][0].lower())  # Glove
                # get the embedding
                bert = bc.encode(li)
                vector = np.concatenate((glove_vector, bert), axis=None)
            Data1.append(vector)
        Feat_Doc.append(Data1)
    np.save('Feat_Doc.npy', Feat_Doc)

## Feature Extraction using BERT and GLOVE for Qn
an = 0
if an == 1:
    bc = BertClient(ip="SERVER_IP_HERE")
    glove_filename = 'glove.6B.50d.txt'
    # glove_path = os.path.abspath(os.path.join('GLOVE', glove_filename))
    word2vec_output_file = glove_filename + '.word2vec'
    # glove2word2vec(glove_path, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    data = np.load('Qn.npy', allow_pickle=True)
    Qn_vector = []
    for i in range(5, 8):  # for all 3 dataset
        tfidf_vector = []
        sin_data = data[i]
        sin_data = [x for x in sin_data if pd.isnull(x) == False]
        sin_data = list(filter(None, sin_data))
        glove_vector = model.get_vector(sin_data[0][0].lower())  # Glove

        # get the embedding
        bert = bc.encode(sin_data)
        vector = np.concatenate((glove_vector, bert), axis=None)

        Qn_vector.append(vector)
    np.save('Qn_vector.npy', Qn_vector)


## Word to Vector for Qn
an = 0
if an == 1:
    data2 = np.load('Qn.npy', allow_pickle=True)
    W2v_qn = []
    for i in range(5, 8):  # for all 3 dataset
        sin_data = data2[i]  # All Questions for first dataset
        sin_data = [x for x in sin_data if pd.isnull(x) == False]
        sin_data = list(filter(None, sin_data))
        # iterate through each sentence in the file
        Dat1 = []
        for z in range(len(sin_data)):
            dat = []
            v = []
            for m in sent_tokenize(sin_data[z]):
                temp1 = []
                # tokenize the sentence into words
                for n in word_tokenize(m):
                    temp1.append(n.lower())
                dat.append(temp1)
                # Create CBOW model
            model = gensim.models.Word2Vec(dat, min_count=1, size=200,
                                           window=5, sg=1)  # Word2vector
            v = model.wv.vectors
            Dat1.append(v)
        W2v_qn.append(Dat1)
    np.save('W2v_qn.npy', W2v_qn)

## similarity between strings
an = 0
if an == 1:
    Input1 = np.load('Data.npy', allow_pickle=True)
    Input2 = np.load('Qn.npy', allow_pickle=True)
    String_SimInd = []
    qn_sim = []
    for i in range(5, 8):  # for all datasets
        string2 = Input2[i]
        string2 = string2.tolist()
        string2 = list(filter(None, string2))
        string2 = [x for x in string2 if pd.isnull(x) == False]
        data_ques = []
        for l in range(len(string2)):  # for all questions in dataset
            val = []
            sim = []
            for j in range(len(Input1[i])):  # for all docs in datasets
                string1 = Input1[i][j]
                if type(string1) == int:
                    con = 0
                    dat = 0
                else:
                    string1 = string1.splitlines()
                    string1 = list(filter(None, string1))
                    for k in range(len(string1)):  # for all strings in the document
                        print(i, j, k)
                        v2 = rem_punct(string2[l])
                        v1 = rem_punct(string1[k])
                        v2.lower();v1.lower()
                        sp = v2.split()
                        cnt = 0
                        for n in range(len(sp)):
                            if sp[n] in v1:
                                cnt = cnt + 1
                        sim.append(cnt)
                        inde = np.zeros((2))
                        inde[0] = j
                        inde[1] = k  # j k
                        val.append(inde)

            max_value = max(sim)
            max_index = sim.index(max_value)
            data_ques.append(val[max_index])
        qn_sim.append(data_ques)
    String_SimInd.append(qn_sim)
    np.save('String_SimInd.npy', String_SimInd)

## Data formation
an = 0
if an == 1:
    pca = PCA(n_components=1)
    data1 = np.load('Feat_Doc.npy', allow_pickle=True)  #  Doc
    data2 = np.load('Qn_vector.npy', allow_pickle=True)  # TFIDF Qn
    data3 = np.load('String_SimInd.npy', allow_pickle=True)  # Similarity indices
    data4 = np.load('W2v_qn.npy', allow_pickle=True)  # W2V Qn
    tot_inputs = []
    tot_targ = []
    for i in range(len(data1)):
        Inputs = data2[i]
        input = np.zeros((len(Inputs), 200))
        targ = np.zeros(len(Inputs))
        for j in range(len(Inputs)):
            p1 = pca.fit_transform(data4[i][j].transpose())
            vect = p1.reshape(1, -1)
            input[j, :] = vect
            ind = data3[0][i][j]
            targ[j] = euclidean_distance(input[j, :].transpose(), data1[i][int(ind[0])])
        tot_inputs.append(input)
        tot_targ.append(targ)
    np.save('Tot_Input.npy', tot_inputs)
    np.save('Tot_Target.npy', tot_targ)



## Optimization For Feature Selection and Transfer learning with Improved LSTM
an = 0
if an == 1:
    bests = []
    Feats_all = np.load('Tot_Input.npy', allow_pickle=True)
    target_all = np.load('Tot_Target.npy', allow_pickle=True)
    bestsol = [];fits = [];train_net = []
    for u in range(3):  # 3 dataset
        print(u+1)
        Feat = Feats_all[u]
        Tar = target_all[u]

        Global_vars.Feat = Feat
        Global_vars.Target = Tar

        # Train the Network for set the min and max weight limit
        hn = 30    # Hidden neuron
        per = round((Feat.shape[0])*(0.70))   # For Training
        train_data = Feat[0:per, :]
        train_target = Tar[0:per]
        test_data = Feat[per:Tar.shape[0]-1,:]
        test_target = Tar[per:Tar.shape[0]-1]
        nett = train_lstm(train_data,train_target, test_data, hn)   # Train the network
        weight = nett.get_weights()
        data_struct = [weight[0].shape, weight[1].shape, weight[2].shape, weight[3].shape, weight[4].shape]
        Global_vars.Stucture = data_struct

        one = data_struct[0][0] * data_struct[0][1]
        two = data_struct[1][0] * data_struct[1][1]
        three = data_struct[2][0]
        four = data_struct[3][0] * data_struct[3][1]
        five = data_struct[4][0]

        train_net.append(nett)

        # Initialize the Parameters
        Npop = 10  # population size
        ch_len = 5 + one + two + three + four + five + 1 # 5 for Feature Selection # one to Five for weight in LSTM and # 1 for  epoches in LSTM
        weight_min = np.min(
            [np.min(weight[0]), np.min(weight[1]), np.min(weight[2]), np.min(weight[3]),
             np.min(weight[4])])  # minimum weight
        weight_max = np.max(
            [np.max(weight[0]), np.max(weight[1]), np.max(weight[2]), np.max(weight[3]),
             np.max(weight[4])])  # maximum weight

        xmin = matlib.repmat(np.concatenate([np.zeros((5)),
                                             weight_min * np.ones((one + two + three + four + five)), 50], axis=None),
                             Npop, 1)
        xmax = matlib.repmat(np.concatenate(
            [Feat.shape[1] * np.ones((5)),
             weight_max * np.ones((one + two + three + four + five)), 100], axis=None), Npop, 1)

        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objective_Function  # objective function
        Max_iter = 25

        print("SFO+WOA...")
        [bestfit1, fitness1, bestsol1, time1] = SFO_WOA(initsol, fname, xmin, xmax, Max_iter)  # SFO

        print("Improved DHOA...")
        [bestfit2, fitness2, bestsol2, time2] = IMP_DHOA(initsol, fname, xmin, xmax, Max_iter)  # Improved DHOA

        print("AOX...")
        [bestfit3, fitness3, bestsol3, time3] = AOX(initsol, fname, xmin, xmax, Max_iter)  # AOX

        print("ABC...")
        [bestfit4, fitness4, bestsol4, time4] = ABC(initsol, fname, xmin, xmax, Max_iter)  # ABC

        print("ABC...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # ABC+AOX


        bs = [bestsol1,bestsol2,bestsol3,bestsol4,bestsol5]

        bestsol.append(bs)

    np.save('bests.npy', bestsol)
    np.save('train_net.npy', train_net)


## Prediction
an = 0
if an == 1:
    bestsols = np.load('bests.npy', allow_pickle=True)
    Feats = np.load('Tot_Input.npy', allow_pickle=True)
    targets = np.load('Tot_Target.npy', allow_pickle=True)
    err_meas= np.zeros((3,8,8))
    net_others = []
    for d in range(3):  # 3 dataset
        Feat = Feats[d]
        Tar = targets[d]

        per = round((Feat.shape[0]) * (0.30))


        for a in range(5):  # For Different Algorithms
            sol = bestsols[d][a]
            selected_Feat = Feat[:, sol[:5].astype('int')]
            train_data = selected_Feat[:per, :]
            train_target = Tar[:per]
            test_data = selected_Feat[per:Feat.shape[0] - 1, :]
            test_target = Tar[per:Feat.shape[0] - 1]
            hn = 30  # Hidden neuron
            nett = train_lstm(train_data, train_target, test_data, hn)  # Train the network
            weight = nett.get_weights()
            data_struct = [weight[0].shape, weight[1].shape, weight[2].shape, weight[3].shape, weight[4].shape]
            Global_vars.Stucture = data_struct
            err_meas[d,a,:] = Modified__Model_LSTM(train_data, train_target, test_data, test_target, sol,Feat,Tar)
        train_data = Feat[:per, :]
        test_data = Feat[per:, :]
        err_meas[d, 5, :] = Model_NN(train_data, train_target, test_data, test_target)
        err_meas[d, 6,:] = Model_DNN(train_data, train_target, test_data, test_target)
        err_meas[d, 7,:] = Model_RNN(train_data, train_target, test_data, test_target)
        err_meas[d, 8, :] = Model_LSTM(train_data, train_target, test_data, test_target)

        err_meas[d, 9, :] = err_meas[d,4,:]
    np.save('err_meas_f.npy',err_meas)


