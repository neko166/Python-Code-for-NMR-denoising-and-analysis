
import tensorflow as tf
from keras.layers import (Dense,Conv1D,BatchNormalization,GlobalAveragePooling1D
                          ,MaxPooling1D,AveragePooling1D,Multiply,Add)





#残差ブロック部分の処理を宣言
class ResidualBlock(tf.keras.layers.Layer):
  #コンストラクタを宣言、フィルター数、カーネルサイズは任意で指定する。ストライドは1で固定
    def __init__(self,filters,kernel_size,strides=1,**kwargs):
        super(ResidualBlock,self).__init__()
        self.conv1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.bn2 = BatchNormalization()
        self.conv3 = Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.bn3 = BatchNormalization()
        self.conv4 = Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.bn4 = BatchNormalization()


        #ショートカット接続のために1次元畳み込みレイヤーを挟んでフィルター数をそろえる
        self.dense_SE1=tf.keras.layers.Dense(filters,activation="relu")
        self.dense_SE2=tf.keras.layers.Dense(filters,activation="sigmoid")
        self.conv1x1=Conv1D(filters=filters,kernel_size=1,strides=strides,padding="same",activation="relu")
        self.global_average_pooling=GlobalAveragePooling1D()



    def call(self,inputs):#コンストラクタで宣言した関数を使って残差接続のサブルーチンを設計する
        shortcut=self.global_average_pooling(inputs)
        shortcut=self.dense_SE1(shortcut)#残差接続する分を控えておく
        shortcut=self.dense_SE2(shortcut)
        reshaped_inputs=self.conv1x1(inputs)
        shortcut=Multiply()([shortcut,reshaped_inputs])


        x = self.conv1(inputs)#最初の畳み込み
        x = self.bn1(x)#バッチ正規化
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x += shortcut#ショートカット接続

        return x




class NoiseGate(tf.keras.layers.Layer):#入力のノイズレベルを評価するよ！
    def __init__(self,kernel_size,strides=1,**kwargs):
        super(NoiseGate,self).__init__()
        self.conv1=Conv1D(filters=16,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.conv2=Conv1D(filters=32,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.conv3=Conv1D(filters=64,kernel_size=kernel_size,strides=strides,padding="same",activation="relu")
        self.conv4=Conv1D(filters=1,kernel_size=kernel_size,strides=strides,padding="same",activation="sigmoid")


    def call(self,inputs):
        x=self.conv1(inputs)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        return x


class MyResNetModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super(MyResNetModel,self).__init__()#記法の意味が分からない…。親クラスの継承がうまくいくおまじない
        #フィルター数を段階的に増やしながら残差ブロックのインスタンスを生成
        self.average_pooling=AveragePooling1D(pool_size=2,padding="valid")
        self.max_pooling=MaxPooling1D(pool_size=2,padding="valid")
        self.residual_block1=ResidualBlock(filters=32,kernel_size=5)#ResidualBlock classのコンストラクタを呼び出している
        #self.residual_block2=ResidualBlock(filters=64,kernel_size=5)
        #self.residual_block3=ResidualBlock(filters=64,kernel_size=5)
        self.residual_block4=ResidualBlock(filters=64,kernel_size=5)
        self.residual_block5=ResidualBlock(filters=128,kernel_size=5)
        self.residual_block6=ResidualBlock(filters=256,kernel_size=5)

   

        self.GAP=GlobalAveragePooling1D()
        #self.dense1 = Dense(8192,activation="relu")
        self.dense2 = Dense(100,activation="relu")
        self.dense3 = Dense(200,activation="relu")
        self.dense4 = Dense(310,activation="linear")

        self.noise_gate=NoiseGate(kernel_size=5)


    def call(self,inputs):

        x = self.residual_block1(inputs)
        x = self.average_pooling(x)
        x = self.residual_block4(x)

        x = self.max_pooling(x)
        x = self.residual_block5(x)
        map = self.residual_block6(x)
      

        gap=self.GAP(map)

        x = self.dense2(gap)
        x = self.dense3(x)
        x = self.dense4(x)


        noise_gate=self.noise_gate(inputs)

        noise_gate = tf.squeeze(noise_gate, axis=-1)  # (batch_size, length)
        inputs = tf.squeeze(inputs, axis=-1)  # (batch_size, length)

        output1=Multiply()([noise_gate, inputs])
        output2=Multiply()([1-noise_gate, x])
        x=Add()([output1,output2])
        

        return x,gap,map


    def get_config(self):
        config = super(MyResNetModel, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



