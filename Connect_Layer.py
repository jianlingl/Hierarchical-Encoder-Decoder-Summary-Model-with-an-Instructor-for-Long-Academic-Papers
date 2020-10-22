# Python3

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
# tf.reset_default_graph()


class Connect_Layer(tf.layers.Layer):
    """定义全连接层"""

    def __init__(self, hidden_size, bias=True, activation=None):
        super().__init__()  # 初始化父类
        self.hidden_size = hidden_size  # 保存配置
        self.bias = bias
        self.activation = activation

    def build(self, input_shape):
        self.W1 = self.add_variable(
            "weight1", [input_shape[-1], self.hidden_size],
            dtype=self.dtype, initializer=tf.glorot_normal_initializer())  # 添加全连接层核
        self.W2 = self.add_variable(
            "weight2", [input_shape[-1], self.hidden_size],
            dtype=self.dtype, initializer=tf.glorot_normal_initializer())  # 添加全连接层核
        if self.bias:
            self.bias = self.add_variable(
                "bias", [self.hidden_size], dtype=self.dtype,
                initializer=tf.glorot_normal_initializer())
        self.built = True  # 说明bulid方法已经被调用

    def call(self, input1,input2):
        tmp1 = tf.matmul(input1, self.W1)
        tmp2 = tf.matmul(input2, self.W2)
        tmp=tmp1+tmp2
        if self.bias:  # 判断是否要使用bias
            tmp += self.bias
        if self.activation:  # 判断是否使用激活函数
            tmp = self.activation(tmp)
        return tmp


# if __name__ == "__main__":
#     x = tf.reshape(tf.range(27, dtype=tf.float32), (9, 3))
#     dense1 = Connect_Layer(5, True, tf.nn.relu)
#     dense2 = Connect_Layer(4, True)
#     y1 = dense1(x,x)
#     y2 = dense2(x,x)
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     y1_value, y2_value = sess.run([y1, y2])
#     print(y1_value)
#     print(y2_value)
