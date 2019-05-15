from BPNN import BP_Network
import random
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def Train_x12(nums):
    def fun1(x):
        y = 0
        for i in range(len(x)):
            y += x[i]
        return y

    # 随机产生训练数据集
    training_sets = []
    for i in range(1000):
        training_inputs = [random.random() / 2, random.random() / 2]
        training_outputs = [fun1(training_inputs)]
        training_sets.append([training_inputs, training_outputs])

    net2 = BP_Network(2, 15, 1, 0.5)
    error_sets = []
    for i in range(nums):
        random.shuffle(training_sets)
        each = training_sets[0:100]
        net2.trains(each)
        error = net2.get_etotal(training_sets)
        error_sets.append(error)
        print(i, error)

    test_sets = []
    for i in range(1000):
        x1 = random.random() / 2
        x2 = random.random() / 2
        y = x1 + x2
        test_sets.append([[x1, x2], [fun1([x1, x2])]])
    # [[x, x], [fun1([x, x])]]
    for x in np.arange(0, 0.5, 0.001):
        tout = []
    outputs = []
    sum = 0
    for i in range(len(test_sets)):
        res = net2.get_result(test_sets[i][0])
        outputs.append(res)
        tout.append((test_sets[i][1]))

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 0.5, 0.01)
    Y = np.arange(0, 0.5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = X + Y
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    plt.savefig("recall.jpg")
    fig = plt.figure()
    ax = Axes3D(fig)
    X = []
    Y = []
    Z = outputs
    for i in range(len(test_sets)):
        X.append(test_sets[i][0][0])
        Y.append(test_sets[i][0][1])
    ax.scatter(X, Y, Z)
    plt.savefig('generation.jpg')
    plt.figure()
    x_err=[]# = np.arange(0, 1, 1 / 100)
    for i in range(nums):
        x_err.append(i)
    plt.plot(x_err, error_sets, 'r-')
    plt.savefig("train.jpg")
# Train_sin(500)
# Train_x12(500)