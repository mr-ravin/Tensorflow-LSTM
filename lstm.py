import tensorflow as tf
import numpy as np

state_size=100 # size of hidden state vector of LSTM.
batch_size=2   # number of batches.
sequence_len=5 # length of sequence to be taken by LSTM.
embedding_len=4 # length of embedding used.

input_data=tf.placeholder(tf.float32,[batch_size,sequence_len,embedding_len]) # input data dimension: 
probabilities=[]                                                              # batch_size x sequence_len x embedding_len
loss=0.0

lstm = tf.contrib.rnn.BasicLSTMCell(state_size)  # state information
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
cell = tf.contrib.rnn.MultiRNNCell([drop]) # cell information along with state information
"""cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers) """ 


initial_state = cell.zero_state(batch_size, tf.float32)  # initial cell-state of LSTM is defined.
outputs, final_st = tf.nn.dynamic_rnn(cell,input_data,initial_state=initial_state)  # auto iterates over states and provide the final state, 
                                                                                    # and the output.


## implementing session

tmp_data=np.array([[[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0]],[[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]]]) # input data defined.

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output_list,final_state=sess.run([outputs,final_st],feed_dict={ input_data:tmp_data}) # prints the final output
  # output_list will have all the outputs , so to get final output produced by LSTM ... use  output_list [batch_number][-1].
  # here [0] represent the first batch, and [-1] represents the last output of that batch.
  print(output_list[0][-1]) 
  # here [1] represent the second batch, and [-1] represents the last output of that batch.
  print(output_list[1][-1])
  # complete list of output
  print("\n**********\n", output_list)
