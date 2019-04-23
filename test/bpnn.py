import numpy as np


class BPNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设定输入层、隐藏层、输出层的节点数nodes、学习率
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        # 为了方便理解，在此将输入设置为mnist数据实例，方便理解各个数据的shape
        # 则input_nodes=784,hidden_nodes=32, output_nodes=64
        # 设定权重值
        # w_in2hid.shape=(32,784)
        self.w_in2hid = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.input_nodes))
        # w_hid2out.shape=(64,32)
        self.w_hid2out = np.random.normal(0.0, self.output_nodes ** -0.5, (self.output_nodes, self.hidden_nodes))
        # 激活函数(logistic函数)
        self.act_func = (lambda x: 1 / (1 + np.exp(-x)))

    def train(self, inputs_org, groundtruth_org):
        # 将输入转化为2d矩阵，输入向量的shape为[feature_dimension,1]
        # input.shape=(784,1)
        inputs = np.array(inputs_org, ndmin=2).T
        # groundtruth.shape=(64,1)
        groundtruth = np.array(groundtruth_org, ndmin=2).T
        # 前向传播
        # hid_ints.shape=(32,1)
        hid_ints = np.dot(self.w_in2hid, inputs)
        # hid_outs.shape=(32,1)
        hid_outs = self.act_func(hid_ints)
        # 输出层（激活函数设置为f(x) = x）
        # out_ints.shape=(64,1)
        out_ints = np.dot(self.w_hid2out, hid_outs)
        # out_outs.shape=(64,1)
        out_outs = out_ints
        # 反向传播
        # out_error.shape=(64,1)
        out_error = out_outs - groundtruth
        # hid_error.shape=(1,32)
        hid_error = np.dot(out_error.T, self.w_hid2out) * (hid_outs * (1 - hid_outs)).T
        # 上式中((1,64).(64,32))*((32,1)*(32,1)).T=(1,32)
        # 更新权重
        # 更新w_hid2out
        self.w_hid2out += out_error * hid_outs.T * self.lr  # shape=(64,32)
        self.w_in2hid += (inputs * hid_error * self.lr).T  # shape=(32,784)\

    def run(self, inputs_org):
        inputs = np.array(inputs_org, ndmin=2).T
        # 实现前向传播
        hid_ints = np.dot(self.w_in2hid, inputs)
        hid_outs = self.act_func(hid_ints)
        # 输出层
        out_ints = np.dot(self.w_in2hid, hid_outs)
        out_outs = out_ints

        return out_outs


if __name__ == '__main__':
    bpnn = BPNetwork()
