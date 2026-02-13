import numpy as np
import matplotlib.pyplot as plt

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

neuron = xw0 + xw1 + xw2 + b

relu = np.max(neuron, 0)

dvalue = 1
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_bias = 1

drelu = dvalue * 1 if relu > 0 else 0
drelu_dxw0 = drelu * dsum_dxw0
drelu_dxw1 = drelu * dsum_dxw1
drelu_dxw2 = drelu * dsum_dxw2
drelu_bias = drelu * dsum_bias

#partial dericatives for multiplication
dmult_x0 = w[0]
dmult_w0 = x[0]
dmult_x1 = w[1]
dmult_w1 = x[1]
dmult_x2 = w[2]
dmult_w2 = x[2]

drelu_x0 = drelu_dxw0 * dmult_x0
drelu_w0 = drelu_dxw0 * dmult_w0

drelu_x1 = drelu_dxw1 * dmult_x1
drelu_w1 = drelu_dxw1 * dmult_w1

drelu_x2 = drelu_dxw2 * dmult_x2
drelu_w2 = drelu_dxw2 * dmult_w2

print(drelu)
print(drelu_dxw0)
print(drelu_dxw1)
print(drelu_dxw2)
print(drelu_bias)

print(drelu_x0)
print(drelu_w0)
print(drelu_x1)
print(drelu_w1)
print(drelu_x2)
print(drelu_w2)
