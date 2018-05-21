from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("./data/mnist/", one_hot=False)
x_train = mnist.train.images
y_train = mnist.train.labels


def fc_layer(bottom,n_weight,name,reuse_flag):
    """

    :param bottom: 网络层的输入值
    :param n_weight: 超参数的第二维的值
    :param name: 该层的名字
    :return:
    """
    n_prev_weight = bottom.get_shape()[1] #获得weight的第一个
    init = tf.truncated_normal_initializer(stddev=0.01)
    with tf.variable_scope("fc_layer",reuse=reuse_flag):
        W = tf.get_variable(name= name+'W',dtype=tf.float32,shape=[n_prev_weight,n_weight],initializer=init)
        b = tf.get_variable(name= name+'b',dtype=tf.float32,initializer=tf.constant(0.01,dtype=tf.float32,shape=[n_weight]))
    fc = tf.nn.bias_add(tf.matmul(bottom,W),b)
    return fc

def network1(x1,reuse_flag):
    fc1 = fc_layer(x1,100,"fc1",reuse_flag)
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1,10,"fc2",reuse_flag)
    return fc2
def network2(x2,reuse_flag):
    fc1 = fc_layer(x2, 100, "fc1", reuse_flag)
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, 10, "fc2", reuse_flag)
    return fc2
def get_same_collection(x_train,y_train):
    x_list = []
    y_list = []
    for number in range(10):
        for i in range(len(y_train)):
            if y_train[i] == number:
                x_list.append(x_train[i])
                y_list.append(number)
    return np.array(x_list),np.array(y_list)
def compute_cost(y_,x1,x2):
    margin = 5.0
    labels_f = y_
    o1 = network1(x1,False)
    o2 = network2(x2,True)
    labels_t = tf.subtract(1.0, y_, name="1-yi")  # labels_ = !labels;
    E_w_2 = tf.pow(tf.subtract(o1, o2), 2)
    E_w_2 = tf.reduce_sum(E_w_2, 1)
    E_w = tf.sqrt(E_w_2 + 1e-6, name="E_w")
    Q = tf.constant(margin, name="Q")
    pos1 = tf.multiply(labels_t, tf.divide(tf.constant(2.0), Q))
    pos = tf.multiply(pos1, E_w_2)
    neg1 = tf.multiply(labels_f, tf.multiply(tf.constant(2.0), Q))
    neg2 = tf.exp(-tf.multiply(tf.divide(tf.constant(2.77), Q), E_w))
    neg = tf.multiply(neg1, neg2)
    # neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(Q, E_w), 0), 2))
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss
def get_result(X1, X2):
    result_1 = network1(X1, True)  # 10 * 样本数
    result_2 = network2(X2, True)
    # print(result_1,result_2)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(result_1 - result_2), 1)) #预测结果,如果两张图片相同，那么他们的值会接近于0 ，如果两种图片不同，那么他们的值会很大
    return E_w
def nn_model(x_train,learning_rate,num_epochs,minibatch_size):
    (m, n_x) = x_train.shape  # m 是样本数，n_x是图片总的像素数
    x1 = tf.placeholder(tf.float32,[None,n_x],name='x1')
    x2 = tf.placeholder(tf.float32,[None,n_x],name='x2')
    y_ = tf.placeholder(tf.float32,[None],name='y_')
    loss = compute_cost(y_,x1,x2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    test_result = get_result(x1, x2)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            x_1, y1 = mnist.train.next_batch(minibatch_size)
            x_2, y2 = mnist.train.next_batch(minibatch_size)

            x_11,y11 = get_same_collection(x_1,y1)
            x_22,y22 =get_same_collection(x_2,y2)
            y_s = np.array(y11 != y22, dtype=np.float32)
            _, minibatch_cost2 = sess.run([optimizer, loss],
                                          feed_dict={x1: x_11, x2: x_22,y_: y_s})
            if epoch % 100 ==0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost2))
        for i in range(100):
            true_count = 0
            test_X1, test_Y1 = mnist.test.next_batch(minibatch_size)
            test_X2, test_Y2 = mnist.test.next_batch(minibatch_size)
            test_Y = np.array(test_Y1 != test_Y2, dtype=np.float32)
            result = sess.run(test_result, feed_dict={x1: test_X1, x2:test_X2})
            for i in range(len(result)):
                if result[i] < 1 and test_Y[i] == 0 or result[i] > 1 and test_Y[i] ==1:
                    true_count += 1
            accuracy = true_count / len(result)
            print(accuracy)
            # print(test_Y)
            # print(result)

nn_model(x_train,learning_rate=0.01,num_epochs=20000,minibatch_size=64)





