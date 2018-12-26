#rnn predict
def next_batch(training_set,batch_set,step):
    rand_start=np.random.randint(0,len(training_data)-steps)
    
    y_batch=np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)


import pandas as pd
milk=pd.read_csv("milk.csv",index_col='Month')
train_set=milk.head(156)
test_set=milk.trail(12)
print (milk)
from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
train_scaled=Scaler.fit_tranmsform(train_set)
test_scaled=Scaler.tranmsform(test_set)

import tensorflow as tf

num_inputs =1

num_time_steps=12

num_neurons=100

num_outputs=1

learning_rate=0.03

num_train_iteration=4000

batch_size=1
x=tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y=tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])

loss=tf.reduce_mean(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

saver=tf.train.Saver()


with tf.Session as sess:
    sess.run(init)
    for iteration in range(num_train_iteration):
        x_batch,y_batch=next_batch(train_scaled,batch_size,num_time_steps)
        sess.run(train,feed_dict={x:x_batch,y:y_batch})
        
        if iteration%100==0:
            mse=loss.eval(feed_dict={x:x_batch,y:y_batch})
        print(iteration,mse)
        
    saver.save(sess,"./ex_time_series_model")
            
    