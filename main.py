import numpy as np


def read(path):
    buf = []
    result = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.split(" ")
            line.pop()
            line = [float(i) for i in line]
            line[-1] = int(line[-1])
            tmp = [0, 0, 0]
            tmp[line[-1]] = 1
            result.append(tmp)
            line.pop()
            buf.append(line)
    return buf, result


def initialize(num_input, num_hidden, num_output, i):
    global learnrate
    global threshold_hidden
    global threshold_output
    global weight_i2h
    global weight_h2o

    learnrate = float(i)
    # threshold_hidden[1][num_hidden]
    threshold_hidden = 2 * np.random.random((1, num_hidden)) - 1
    # threshold_output[1][num_output]
    threshold_output = 2 * np.random.random((1, num_output)) - 1
    # weight_i2h[num_input][num_hidden]
    weight_i2h = 2 * np.random.random((num_input, num_hidden)) - 1
    # weight_h2o[num_hidden][num_output]
    weight_h2o = 2 * np.random.random((num_hidden, num_output)) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train(traindata, trainresult, threshold_hidden, threshold_output, weight_i2h, weight_h2o):
    for i in range(len(traindata)):
        # input[1][num_input]
        input = np.mat(traindata[i]).astype(float)
        # output[1][num_output]
        output = np.mat(trainresult[i]).astype(float)

        # 前向计算
        # step1 隐含层值
        # hidden_in[1][num_hidden]
        hidden_in = np.dot(input, weight_i2h).astype(float)  # ah
        # hidden_out[1][num_hidden]
        hidden_out = sigmoid(hidden_in - threshold_hidden).astype(float)  # bh
        # step2 输出层值
        # output_in[1][num_output]
        output_in = np.dot(hidden_out, weight_h2o).astype(float)
        # output_out[1][num_output]
        output_out = sigmoid(output_in - threshold_output).astype(float)

        # 求误差传播参数
        # g[1][num_output]
        g = np.multiply(np.multiply(output_out, 1 - output_out).astype(float), output - output_out)

        # b[1][num_hidden]
        # g[1][num_output] * np.transpose(weight_h2o)[num_output][num_hidden] ---->    b1[1][num_hidden]
        # e[1][num_hidden]
        b = np.multiply(1 - hidden_out, hidden_out)
        b1 = np.dot(g, np.transpose(weight_h2o))
        e = np.multiply(b, b1)

        # Updata
        threshold_hidden -= learnrate * e
        threshold_output -= learnrate * g
        # weight_h2o[num_hidden][num_output]  ---->  [num_hidden][1] * [1][num_output]
        # learnrate * gj * bh
        weight_h2o += learnrate * np.dot(np.transpose(hidden_out), g)
        # weight_i2h[num_input][num_output]  ---->  [num_input][1] * [1][num_hidden]
        # learnrate* xi * eh
        weight_i2h += learnrate * np.dot(np.transpose(input), e)


def test(testdata, testresult, threshold_hidden, threshold_output, weight_i2h, weight_h2o):
    accuracy = 0
    for i in range(len(testdata)):
        input = np.mat(testdata[i]).astype(float)
        output_expected = np.mat(testresult[i]).astype(float)
        hidden_in = np.dot(input, weight_i2h).astype(float)
        hidden_out = sigmoid(hidden_in - threshold_hidden).astype(float)
        output_in = np.dot(hidden_out, weight_h2o).astype(float)
        output_out = sigmoid(output_in - threshold_output).astype(float)
        index = np.argmax(output_out, axis=1)
        if output_expected[0, index] == 1:
            accuracy += 1
    return accuracy / len(testdata)


if __name__ == "__main__":
    traindata = read("Iris-train.txt")
    testdata = read("Iris-test.txt")
    LearnRate = 0.01
    # initialize(4, 10, 3, LearnRate)
    accuracy_list = []
    for i in range(10):
        initialize(4, 10, 3, LearnRate)
        for j in range(1000):
            train(traindata[0], traindata[1], threshold_hidden, threshold_output, weight_i2h, weight_h2o)
        acc = test(testdata[0], testdata[1], threshold_hidden, threshold_output, weight_i2h, weight_h2o)
        print("the accuracy of test{} is {:.3}%".format(i + 1, acc * 100.0))
        accuracy_list.append(acc)
    avg = np.mean(accuracy_list)
    std = np.std(accuracy_list)
    print("Average accuracy is {:.4}% and standard deviation is {:.3}".format(avg * 100, std))
