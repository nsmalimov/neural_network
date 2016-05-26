# -*- coding: utf-8 -*- 

import neural_networks
from math import exp
from scipy import optimize
import time
from pylab import *

#функция активации
def activation_func(summator):
    return 1/(1 + exp(-summator))

#матрица (схема) нейронной сети, каждый её элемент абстрагирует нейрон
#принимает количество слоёв и количество нейронов в слое
def create_matrix(num_layers, num_neuron):
	matrix_dist = []
	help_array = []
	for i in xrange(num_layers):
            for j in xrange(num_neuron):
                help_array.append(1)
	    matrix_dist.append(help_array)
            help_array = []
	return matrix_dist

#суммируем и пропускаем через функцию активации 
#таким образом, получаем выход нужного нейрона
def one_neuron_out(x_array, w_array, parametr):
    #суммирование
    summator = 0
    for i in xrange(len(x_array)):
    	summator = summator + x_array[i]*w_array[i]
    if (parametr == 0):
       return activation_func(summator)
    if (parametr == 1):
        return summator


#классифицируем
#последний скрытый слой
def last_neuron(x_array):
    count_less = 0
    count_more = 0
    for i in x_array:
        if (i > 0.7):
            count_more +=1
        if (i <= 0.7):
            count_less +=1
    if (count_more >= count_less):
        return 1.0
    if (count_more < count_less):
        return 0.0

#проход вперёд по сети
def classificate_one(matrix_dist, x_array, w_matrix, parametr):
    last_layer = []
    result_neuron_matrix = []
    for i in xrange(num_layers ):
        result_neuron_matrix.append([])
    #веса константы
    #начальное значение вектора x
    last_layer = x_array 
    last_layer_help = []
    #вычисляем по слоям
    for i in xrange(len(matrix_dist)):
        #вычисляем по каждому нейрону в слое
        for j in xrange(len(matrix_dist[0])):
            answer_neuron = one_neuron_out(last_layer, w_matrix[i][j], 0)
            result_neuron_matrix[i].append(answer_neuron)
            last_layer_help.append(answer_neuron)
        last_layer = last_layer_help
        last_layer_help = []
    if parametr == 0:
        return last_neuron(last_layer)
    if parametr == 1:
        return last_layer, result_neuron_matrix

#вычислим ошибку
def count_error(true_answer, our_answer):
    return (true_answer - our_answer) * (-1) * our_answer * (1 - our_answer)

#обратное распространение ошибки
def train_one_back(matrix_dist, x_array, true_answer, w_matrix, num_layers, is_shake_up):
    step_error = 0
    true_answer = true_answer + 0.0
    result_front, result_neuron_matrix = classificate_one(matrix_dist, x_array, w_matrix, 1)
    if (is_shake_up == 1):
       w_matrix  = shake_up(w_matrix, matrix_dist, num_layers)
    y_array = [true_answer, true_answer, true_answer, true_answer] 
    error_array = [count_error(y_array[0], result_front[0]), count_error(y_array[1], result_front[1]), count_error(y_array[2], result_front[3]), count_error(y_array[3], result_front[3])]
    
    num = len(w_matrix)-1
    out_array = []
    error_help = []
    #по слоям
    while num >= 0:
        #по нейронам
        error_array_help = []
        for i in xrange(len(w_matrix[num])):  
            answer_neuron = one_neuron_out(error_array, w_matrix[num][i], 1)
            error = count_error(result_neuron_matrix[num][i], answer_neuron)
            for j in xrange(len(w_matrix[num][i])):
                w_matrix[num][i][j] = w_matrix[num][i][j] + 1.0 * result_neuron_matrix[num][i] * error
            error_array_help.append(error)
        error_array = error_array_help
        if (error_array_help[0] < 0):
           step_error = (-1) * error_array_help[0]
        else:
           step_error = error_array_help[0] 
        num -=1
    return w_matrix, step_error

def shake_up(w_matrix, matrix_dist, num_layers):
    w_matrix = []
    for i in xrange(num_layers):
        w_matrix.append([])
        for j in xrange(len(matrix_dist[0])):
            w_matrix[i].append([1,1,1,1])
    return w_matrix

def metrics(test_Y, answer_array):
    a = 0
    b = 0
    c = 0
    d = 0 
    
    for i in xrange(len(test_Y)):
        if (answer_array[i] == 1 and test_Y[i] == 1):
           a +=1
        if (answer_array[i] == 1 and test_Y[i] == 0):
           b +=1
        if (answer_array[i] == 0 and test_Y[i] == 1):
           c +=1
        if (answer_array[i] == 0 and test_Y[i] == 0):
           d +=1

    recall = a / (a + c + 0.0)
    precision = a / (a + b + 0.0)
    accuracy = (a + d) / (a + b + c + d + 0.0)
    error = (b + c) / (a + b + c + d + 0.0)
    print a, " ", b, " ", c, " ",d
    print recall, " ", precision, " ", accuracy, " ",error
    
#получаем массивы из набора данных
train_X, train_Y, test_X, test_Y = neural_networks.prepare_data("data_banknote_authentication_rand.txt")

#количество скрытых слоёв
num_layers = 2

#количество нейронов в каждом слое
num_neuron = len(train_X[0]) 


matrix_dist = create_matrix(num_layers, num_neuron)

#послойное представление весов
#до обучения задаём веса некоторыми значениями 
w_matrix = []
for i in xrange(num_layers):
    w_matrix.append([])
    for j in xrange(len(matrix_dist[0])):
        w_matrix[i].append([1,1,1,1])

#обучение на всём наборе обучающей выборки
import time
start = time.clock()
step_error_array = []
is_shake_up = 0
for i in xrange(len(train_X)):
    if (i == len(train_X) / 2):
        is_shake_up = 1
    else: is_shake_up = 0
    w_matrix, step_error = train_one_back(matrix_dist, train_X[i], train_Y[i], w_matrix, num_layers, is_shake_up)
    step_error_array.append(step_error) 
end = time.clock()
print 'Time: %s' % (end - start)
#небольшое исправление набора исходных данных (фильтрация)
test_X[58] = test_X[58][0:4]

#классификация
true_answer = 0
false_answer = 0

answer_array = []

for i in xrange(len(test_X)):
    classif_count = classificate_one(matrix_dist, test_X[i], w_matrix, 0)
    answer_array.append(classif_count)
    if (classif_count == test_Y[i] + 0.0):
       true_answer += 1
    else:
        false_answer += 1

print true_answer, false_answer
