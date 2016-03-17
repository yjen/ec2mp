import parse_log as util
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

log_name = "log/miniplaces.train_log_cla.txt"
log_name2 = "log/miniplaces.train_log_cla_cont.txt"

# a = util.parse_log(log_name)



def parse_log(log_path):
    iteration = []
    accuracy_at_1 = []
    accuracy_at_5 = []
    loss = []
    with open(log_path) as f:
        for line in f:
            if not re.search('Iteration', line):
                continue
            numbers = re.sub('Iteration|:|accuracy_at_[0-9]|=|\n|loss|%|;|','' , line)
            numbers = re.split('\ *', numbers)
            iteration.append(int(numbers[1]))
            accuracy_at_1.append(float(numbers[2]))
            accuracy_at_5.append(float(numbers[3]))
            loss.append(float(numbers[4]))

    plt.plot(iteration, loss)
    plt.show()

def parse_log(log_path1, log_path2):
    iteration = []
    accuracy_at_1 = []
    accuracy_at_5 = []
    loss = []
    with open(log_path1) as f:
        for line in f:
            if not re.search('Iteration', line):
                continue
            numbers = re.sub('Iteration|:|accuracy_at_[0-9]|=|\n|loss|%|;|','' , line)
            numbers = re.split('\ *', numbers)
            iteration.append(int(numbers[1]))
            accuracy_at_1.append(float(numbers[2]))
            accuracy_at_5.append(float(numbers[3]))
            loss.append(float(numbers[4]))

    last_iteration = iteration[-1]
    with open(log_path2) as f:
        for line in f:
            if not re.search('Iteration', line):
                continue
            numbers = re.sub('Iteration|:|accuracy_at_[0-9]|=|\n|loss|%|;|','' , line)
            numbers = re.split('\ *', numbers)
            iteration.append(last_iteration + int(numbers[1]))
            accuracy_at_1.append(float(numbers[2]))
            accuracy_at_5.append(float(numbers[3]))
            loss.append(float(numbers[4]))

    plt.plot(iteration, loss)
    plt.show()

parse_log(log_name, log_name2)
