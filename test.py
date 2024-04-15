import numpy as np

x = np.random.randn(2,256,256)
x = np.argmax(x, 0)
np.int8(x)

color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])
print(color_label.shape, x.shape)
y = color_label[x]
print(y.shape)