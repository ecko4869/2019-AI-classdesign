from BPNN import BP_Network
import random
import numpy as np
import matplotlib.pyplot as plt


def Train_sin(nums):
    training_sets = [[[x / (np.pi)], [np.sin(x)]] for x in np.arange(0, np.pi, 0.01)]

    net3 = BP_Network(1, 10, 1, 0.5)
    error_sets = []
    for i in range(nums):
        # random.shuffle(training_sets)
        each = training_sets[0:315]
        net3.trains(each)
        error = net3.get_etotal(training_sets)
        error_sets.append(error)
        # print(i, error)

    test_sets = [[[x / (np.pi)], [(np.sin(x))]] for x in np.arange(0, np.pi, 0.01)]

    tout = []
    outputs = []
    for i in range(len(test_sets)):
        res = net3.get_result(test_sets[i][0])
        outputs.append(res)
        tout.append((test_sets[i][1]))
    x = np.arange(0, 3.15, 0.01)
    x_err = []
    for i in range(len(error_sets)):
        x_err.append(i)
    # plt.figure()
    # plt.subplot(1, 3, 1)
    plt.plot(x, tout, 'g-', x, outputs, 'b-')

    plt.savefig('recall.jpg')
    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.plot(x_err, error_sets, 'r-')
    plt.savefig('train.jpg')
    test_sets = [[[x / (np.pi)], [(np.sin(x))]] for x in np.arange(0, np.pi, 0.005)]
    tout = []
    outputs = []
    x = np.arange(0, np.pi, 0.005)
    for i in range(len(test_sets)):
        res = net3.get_result(test_sets[i][0])
        outputs.append(res)
        tout.append((test_sets[i][1]))
    # plt.figure()
    # plt.subplot(1, 3, 3)
    plt.figure()
    plt.plot(x, tout, 'g-', x, outputs, 'r-')
    # plt.show()
    plt.savefig('generation.jpg')


# Train_sin(500)
