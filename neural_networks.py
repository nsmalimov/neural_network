# -*- coding: utf-8 -*-


def prepare_data(filename):
    f = open(filename, 'r')
    train_X = []
    train_Y = []

    test_X = []
    test_Y = []

    readlines = f.readlines()
    limiter_train = 1072

    for i in xrange(limiter_train):
        train_X.append([])
        readlines_split = readlines[i].split(";")
        for j in readlines_split:
            if (j == readlines_split[-1]):
                None
            else:
                train_X[i].append(float(j))
        train_Y.append(int(readlines_split[-1][0:len(readlines_split) - 1]))

    count = 0
    for i in xrange(limiter_train - 1, len(readlines)):
        test_X.append([])
        readlines_split = readlines[i].split(";")
        for j in readlines_split:
            if (j == readlines_split[-1]):
                None
            else:
                test_X[count].append(float(j))
        test_Y.append(int(readlines_split[-1][0:len(readlines_split) - 1]))
        count = count + 1
    return train_X, train_Y, test_X, test_Y
