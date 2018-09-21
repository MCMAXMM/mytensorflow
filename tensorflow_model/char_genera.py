import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import unidecode
from tqdm import tqdm
import re
import random
import time
#返回一个下载文件所在的路径例如：C:\Users\pc\.keras\datasets\shakespeare.txt
path_to_file=tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/yashkatariya/shakespeare.txt')
text=unidecode.unidecode(open(path_to_file).read())
print(type(text))
unique=sorted(set(text))
char2idx={u:i for i,u in enumerate(unique)}
idx2char={i:u for i,u in enumerate(unique)}
max_length=100
vocab_size=len(unique)
embedding_dim=256
units=1024
BATCH_SIZE=64
BUFFER_SIZE=10000
input_text=[]
target_text=[]
for f in range(0,len(text)-max_length,max_length):
    inps=text[f:f+max_length]
    targ=text[f+1:f+max_length+1]
    input_text.append([char2idx[x] for x in inps])
    target_text.append([char2idx[x] for x in targ])
print(np.array(input_text).shape)
print(np.array(target_text).shape)

dataset=tf.data.Dataset.from_tensor_slices((input_text,target_text)).shuffle(BUFFER_SIZE)
#下面主要是扔掉不满足batch的最后一部分
#因为batch_size不一定能整除data_size,所以剩下数据
dataset=dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))


class Model(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,units,batch_size):
        super(Model,self).__init__()
        self.units=units
        self.batch_sz=batch_size
        self.embedding=tf.keras.layers.Embedding(vocab_size,embedding_dim)
        if tf.test.is_gpu_available():
            self.gru=tf.keras.layers.CuDNNGRU(self.units,return_sequences=True,
                                return_state=True,recurrent_initializer="glorot_uniform")
        else:
            self.gru=tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True)
        self.fc=tf.keras.layers.Dense(vocab_size)
    def call(self,x,hidden):
        #[batch,maxlength]
        x=self.embedding(x)#[batch,maxlength,dim]
        #output shape=(batch_size,max_length,hidden_size)
        #states shape=(batch_size,hidden_size)返回的是最后一个cell的states
        output,states=self.gru(x,initial_state=hidden)
        output=tf.reshape(output,(-1,output.shape[2]))#[batch_size*max_length,hidden_size]
        x=self.fc(output)#[batch_size*max_length,vocab_size]
        return x,states
model=Model(vocab_size,embedding_dim,units,BATCH_SIZE)
optimizer=tf.train.AdamOptimizer()
# using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
#使用这个sparse_softmax_cross_entropy，我们就不用创建one-hot编码了
def loss_function(real,preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real,logits=preds)
EPOCHS=30
for epoch in range(EPOCHS):
    start=time.time()
    #在每个epoch开始之前重新初始化hidden state
    hidden=model.reset_states()
    for (batch,(inp,target)) in tqdm(enumerate(dataset)):
        with tf.GradientTape() as tape:
            predictions,hidden=model(inp,hidden)
            target=tf.reshape(target,(-1,))
            loss=loss_function(target,predictions)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         loss))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
num_generate = 1000

# You can change the start string to experiment
start_string = 'Q'
# converting our start string to numbers(vectorizing!)
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# empty string to store our results
text_generated = ''

# low temperatures results in more predictable text.
# higher temperatures results in more surprising text
# experiment to find the best setting
temperature = 1.0

# hidden state shape == (batch_size, number of rnn units); here batch size == 1
hidden = [tf.zeros((1, units))]
for i in range(num_generate):
    predictions, hidden = model(input_eval, hidden)

    # using a multinomial distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()

    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated += idx2char[predicted_id]

print(start_string + text_generated)
