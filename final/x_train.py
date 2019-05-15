from BPNN import BP_Network
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Train_x(nums):
    fun = lambda x: x ** 2
    net1 = BP_Network(1, 6, 1, 0.5)
    training_sets = []
    for i in range(100):
        training_inputs = []
        training_outputs = []
        a = random.random()
        # a = 1
        b = fun(a)
        training_inputs.append(a)
        training_outputs.append(b)
        training_sets.append([training_inputs, training_outputs])

    error_sets = []

    for i in range(nums):
        random.shuffle(training_sets)
        each = training_sets[0:90]
        net1.trains(each)
        error = net1.get_etotal(training_sets)
        error_sets.append(error)
        print(i, error)

    test_sets = []
    for i in range(100):
        test_inputs = []
        test_outputs = []
        a = random.random()
        b = fun(a)
        test_inputs.append(a)
        test_outputs.append(b)
        test_sets.append([test_inputs, test_outputs])

    test_sets = [[[x], [fun(x)]] for x in np.arange(0, 1, 0.01)]
    tout = []
    outputs = []
    plt.figure()
    for i in range(len(test_sets)):
        res = net1.get_result(test_sets[i][0])
        outputs.append(res)
        tout.append((test_sets[i][1]))

    x = np.arange(0, 1, 1 / 100)
    plt.plot(x, tout, 'g-', x, outputs, 'b-')
    plt.savefig('recall.jpg')
    x_err = []
    for i in range(nums):
        x_err.append(i)
    plt.figure()
    plt.plot(x_err, error_sets, 'r-')
    plt.savefig('train.jpg')
    plt.figure()
    test_sets = [[[x], [fun(x)]] for x in np.arange(0, 1, 0.001)]
    tout = []
    outputs = []
    x = np.arange(0, 1, 0.001)
    for i in range(len(test_sets)):
        res = net1.get_result(test_sets[i][0])
        outputs.append(res)
        tout.append((test_sets[i][1]))
    plt.plot(x, tout, 'g-', x, outputs, 'r-')
    plt.savefig('generation.jpg')
# Train_sin(500)
# Train_x(500)