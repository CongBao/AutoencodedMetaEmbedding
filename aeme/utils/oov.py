# pre-processing to predict oovs

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import mse

class Regressor(object):
    
    def __init__(self,
                 in_size,
                 out_size,
                 activ_func='sigmoid',
                 batch_size=64,
                 learning_rate=0.001,
                 epoch=100):
        self.in_size = in_size
        self.out_size = out_size
        self.activ_func = activ_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch

    def build(self):
        src = Input(shape=(self.in_size,))
        out = Dense(self.out_size, activation=self.activ_func)(src)
        self.model = Model(src, out)
        self.model.compile(Adam(lr=self.learning_rate), loss=mse)
    
    def train(self, x, y):
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epoch)

    def predict(self, x):
        self.model.predict(x, batch_size=self.batch_size)
