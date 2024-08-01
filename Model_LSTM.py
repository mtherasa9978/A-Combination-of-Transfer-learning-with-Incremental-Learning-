import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluate_Error import evaluate_error




def Model_LSTM(trainX, trainY, testX, test_y):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(trainX, trainY, epochs= 5, batch_size=1, verbose=2)
    testPredict= model.predict(testX).ravel()
    act = test_y.reshape(len(test_y), 1)
    pred = testPredict
    err = evaluate_error(act, pred)
    return err


# def Model_LSTM1(trainX, trainY, testX, test_y):
#     trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
#     model = Sequential()
#     model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='sgd')
#     testPredict = np.zeros((testX.shape[0], trainY.shape[1]))
#     predict = np.zeros((test_y.shape[0], test_y.shape[1]))
#     for i in range(trainY.shape[1]):
#         model.fit(trainX, trainY[:, i], epochs= 2, batch_size=1, verbose=2)
#         testPredict[:, i] = model.predict(testX).ravel()
#     predict = np.round(testPredict)
#     eval = evaluation(predict, test_y)
#     return np.asarray(eval).ravel()

def train_lstm(trainX, trainY, testX, ep):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    # testPredict = np.zeros((testX.shape[0], trainY.shape[1]))
    # predict = np.zeros((test_y.shape[0], test_y.shape[1]))
    # for i in range(trainY.shape[1]):
    #     model.fit(trainX, trainY[:, i], epochs= 2, batch_size=1, verbose=2)
    model.fit(trainX, trainY, epochs=ep, batch_size=1, verbose=2)
    return model




def Modified_Model_LSTM(train_data, train_target, test_data, test_target,sol=None, Structure=None):

    weight = []
    one = train_data.shape[1] * Structure[0][1]
    two = Structure[1][0] * Structure[1][1]
    three = Structure[2][0]
    four = Structure[3][0] * Structure[3][1]
    five = Structure[4][0]
    weight.append(np.reshape(sol[0:one], (train_data.shape[1], Structure[0][1])))
    weight.append(np.reshape(sol[one:one + two], (Structure[1][0], Structure[1][1])))
    weight.append(np.reshape(sol[one + two:one + two + three], (Structure[2][0])))
    weight.append(np.reshape(sol[one + two + three:one + two + three + four],
                             (Structure[3][0], Structure[3][1])))
    weight.append(
        np.reshape(sol[one + two + three + four:one + two + three + four + five], (Structure[4][0])))
    out, model = LSTM_train(train_data, train_target, test_data, weight, sol[-1].astype('int'))

    return out

def Modified__Model_LSTM(train_data, train_target, test_data, test_target,sol=None, Structure=None):

    weight = []
    one = train_data.shape[1] * Structure[0][1]
    two = Structure[1][0] * Structure[1][1]
    three = Structure[2][0]
    four = Structure[3][0] * Structure[3][1]
    five = Structure[4][0]
    weight.append(np.reshape(sol[0:one], (train_data.shape[1], Structure[0][1])))
    weight.append(np.reshape(sol[one:one + two], (Structure[1][0], Structure[1][1])))
    weight.append(np.reshape(sol[one + two:one + two + three], (Structure[2][0])))
    weight.append(np.reshape(sol[one + two + three:one + two + three + four],
                             (Structure[3][0], Structure[3][1])))
    weight.append(
        np.reshape(sol[one + two + three + four:one + two + three + four + five], (Structure[4][0])))
    out, model = LSTM_train(train_data, train_target, test_data, weight, sol[-1].astype('int'))
    act = test_target.reshape(len(test_target), 1)
    pred = out
    err = evaluate_error(act ,pred)
    return err


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, weight, sol):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    # model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    if weight is not None:
        model.set_weights(weight)
    else:
        model.fit(trainX, trainY, epochs=sol[0], batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model