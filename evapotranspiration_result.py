import tensorflow as tf
import numpy as np


Data = np.loadtxt('Jinju-Changwon.csv', delimiter=',', dtype=np.float32)

x_data = Data[10000:, [-1]]    #앞서 Modeling에서 1만번까지 사용했으므로 이후로는 Testing에 사용.

X = tf.placeholder(tf.float32, shape=[None, 1])    #마찬가지로 앞선 Modeling의 x랑 동일하게 맞춰줍니다.
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([1, 1]), name='weight')    #여기서도 [x, 1]로 맞춰줍니다.
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)

    save_path = "./savedSin.cpkt"    #Modeling의 save_path이름과 같은 이름을 지정해서 넣어주면 학습된 자료를 바탕으로 Testing을 진행합니다.
    saver.restore(sess, save_path)

    # data = (x_data,)

    dict = sess.run(hypothesis, feed_dict={X: x_data})
    newList = np.ravel(dict)

    np.savetxt("Jinju-ChangwonResult.csv", newList, delimiter=',')    #결과값을 "Title"로 csv형식으로 만들어 냅니다.