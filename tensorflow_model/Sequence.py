import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')
class Sequence(keras.Model):
    def __init__(self,gru_hidden):
        super(Sequence,self).__init__()
        self.gru_hidden=gru_hidden
        self.gru=gru(self.gru_hidden)
    def call(self,inputs,hidden,att_hid,atten_num):
        output,state=self.gru(inputs,initial_state=hidden)
        output=self.Gru_Self_Attention(output,att_hid,atten_num)
        return output
    def Gru_Self_Attention(self,input,att_hid_dim,atten_num):
        #input 的shape 是：[BATCH,squ_len,item_hid+tat_hid][30,6,54]
        #atten_dim 执行attention的次数
        # att_hid_dim:执行attention Da的维度 看论文
        #H*Ws1
        layer1=tf.layers.dense(input,att_hid_dim,use_bias=False)
        layer2=tf.nn.tanh(layer1)
        #下面会得到n*atten_num
        layer3=tf.layers.dense(layer2,atten_num,use_bias=False)
        #attention weight
        #[BATCH,n,r]
        #r:aspect
        A=tf.nn.softmax(layer3,axis=1)
        #transpose可以不知道形状就可以交换，比reshape好一点
        A=tf.transpose(A,[0,2,1])

        #[BATCH,r,input_hid_dim]
        M=tf.matmul(A,input)#POI那篇的Zu
        M=tf.transpose(M,[0,2,1])
        z_u=tf.layers.dense(M,1)
        z_u=tf.nn.relu(z_u)

        z_u=tf.squeeze(z_u,axis=2)
        #最后融化了attnetion的vector
        #[BATCH,input_hid_dim]输入时隐层的维度
        return z_u
if __name__=="__main__":
    #测试代码
    my_gru=Sequence(64)
    a=tf.ones((12,8,6))
    hidden_inia=tf.zeros((12,64))
    output=my_gru(inputs=a,hidden=hidden_inia,att_hid=10,atten_num=6)
    print(output.shape)
    print(my_gru.summary())
