import tensorflow as tf
import numpy as np

data = np.loadtxt("Jinju-Changwon.csv", delimiter=',', dtype=np.float32)

x_data = data[:10000, 0:-1]    #데이터 번호 1~10,000까지 먼저 지정한다.
y_data = data[:10000, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 1])    #만약 다중선형회귀라면 라벨의 갯수에 맞게 맞춰준다.[None, x]
Y = tf.placeholder(tf.float32, shape=[None, 1])    #y는 증발산량 1개이므로 1

W = tf.Variable(tf.random_normal([1, 1]), name='weight')    #여기서 [x,1] 위의 x랑 맞춰준다.
b = tf.Variable(tf.random_normal([1]), name='bias')    #y는 증발산량 1개이므로 1

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(50001):    #학습횟수
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10000 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)    #1만번 학습할 때 마다 cost, prediction을 출력

saver = tf.train.Saver()
save_path = saver.save(sess, "./savedJinju-Changwon.cpkt")    #학습모델을 저장합니다.
print("학습된 모델을 저장했습니다.")
