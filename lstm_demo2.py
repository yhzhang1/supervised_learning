from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

t1 = 0.1
t2 = 0.2
t3 = 0.3

# define model
#shape=(3,1)代表输入数据的维度，问题是3个输入的数字，一个输出的数字，所以是(3,1）
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
# define input data
#输入数据维度也是(3,1)，因为只有一个，所以整体的data就是(1,3,1)
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))