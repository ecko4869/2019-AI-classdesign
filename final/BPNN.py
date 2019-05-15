import random
import math


# 神经元
class Neuron:
    def __init__(self, b):
        self.biase = b
        self.weight = []
        self.sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))

    def calc_output(self, input):
        self.input = input
        sum = 0
        for i in range(len(input)):
            sum += self.input[i] * self.weight[i]
        sum += self.biase
        self.output = self.sigmoid(sum)
        return self.output

    def calc_error(self, out):
        return 0.5 * (out - self.output) ** 2

    def calc_inputLayer_error(self, out):
        return -(out - self.output) * self.output * (1 - self.output)

    def calc_outputLayer_error(self, out):
        return -(out - self.output)

    # 输出层到输入层的误差
    def error_outputLayer2inputLayer(self):
        return self.output * (1 - self.output)

    # 输入层对权重的误差
    def error_inputLayer2weight(self, i):
        return self.input[i]


class Layer:
    def __init__(self, num, biase):
        if biase == None:
            self.biase = random.random()
        else:
            self.biase = biase
        self.neuron = []
        for i in range(num):
            temp = Neuron(self.biase)
            self.neuron.append(temp)

    def output(self, input):
        outputs = []
        for neuron in self.neuron:
            temp = neuron.calc_output(input)
            outputs.append(temp)
        return outputs


class BP_Network:
    def __init__(self, num_input, num_hidden, num_output, learning_rate, hidden_weight=None, hidden_biase=None,
                 output_weight=None, output_biase=None):
        self.learning_rate = learning_rate
        self.hiddenLayer = Layer(num_hidden, hidden_biase)
        self.outputLayer = Layer(num_output, output_biase)
        tot = 0
        for i in range(num_hidden):
            for j in range(num_input):
                if not hidden_weight:
                    self.hiddenLayer.neuron[i].weight.append(random.random())
                else:
                    self.hiddenLayer.neuron[i].weight.append(hidden_weight[tot])
                tot += 1

        tot2 = 0
        for i in range(num_output):
            for j in range(num_hidden):
                if not output_weight:
                    self.outputLayer.neuron[i].weight.append(random.random())
                else:
                    self.outputLayer.neuron[i].weight.append(output_weight[tot2])
                tot2 += 1

    # 前向传播
    def forward(self, inputs):
        hidden_outputs = self.hiddenLayer.output(inputs)
        final_outputs = self.outputLayer.output(hidden_outputs)
        return final_outputs

    def backward(self, input, output):
        self.forward(input)
        update_output_error = len(self.outputLayer.neuron) * [0]
        for i in range(len(self.outputLayer.neuron)):
            update_output_error[i] = self.outputLayer.neuron[i].calc_inputLayer_error(output[i])

        update_hidden_error = len(self.hiddenLayer.neuron) * [0]
        for i in range(len(self.hiddenLayer.neuron)):
            temp = 0
            for j in range(len(self.outputLayer.neuron)):
                temp += update_output_error[j] * self.outputLayer.neuron[j].weight[i]

            update_hidden_error[i] = temp * self.hiddenLayer.neuron[i].error_outputLayer2inputLayer()

        # 更新权值
        for i in range(len(self.outputLayer.neuron)):
            for j in range(len(self.outputLayer.neuron[i].weight)):
                delta_weight = update_output_error[i] * self.outputLayer.neuron[i].error_inputLayer2weight(j)
                self.outputLayer.neuron[i].weight[j] -= delta_weight * self.learning_rate

        for i in range(len(self.hiddenLayer.neuron)):
            for j in range(len(self.hiddenLayer.neuron[i].weight)):
                delta_weight = update_hidden_error[i] * self.hiddenLayer.neuron[i].error_inputLayer2weight(j)
                self.hiddenLayer.neuron[i].weight[j] -= delta_weight * self.learning_rate

    def trains(self, train_data):
        for i in range(len(train_data)):
            self.backward(train_data[i][0], train_data[i][1])

    def get_etotal(self, train_data):
        sum = 0
        for i in range(len(train_data)):
            train_input, train_output = train_data[i]
            self.forward(train_input)
            for j in range(len(train_output)):
                sum += self.outputLayer.neuron[j].calc_error(train_output[j])
        return sum

    def get_result(self, test_data):
        hidden_output = []
        for i in range(len(self.hiddenLayer.neuron)):
            hidden_output.append(self.hiddenLayer.neuron[i].calc_output(test_data))
        outputs = []
        for i in range(len(self.outputLayer.neuron)):
            outputs.append(self.outputLayer.neuron[i].calc_output(hidden_output))
        return outputs
