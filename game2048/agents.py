from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
import json
import numpy as np
import keras
import random
from keras.datasets import mnist
from keras.models import Sequential,load_model,model_from_json
from keras.optimizers import SGD,RMSprop
from keras.layers import BatchNormalization,Dense, Dropout, Flatten, MaxPooling3D, MaxPooling2D ,Activation ,Concatenate ,Conv3D,Conv2D
from keras.utils import to_categorical
from keras.utils import np_utils
from keras import backend as K

model1 = load_model('2048_128.h5')
model2 = load_model('2048_256.h5')
model3 = load_model('2048_512.h5')
model4 = load_model('2048_1024_1.h5')
model5 = load_model('2048_1024_2.h5')
model6 = load_model('2048_2048_1.h5')
model7 = load_model('2048_2048_2.h5')
model8 = load_model('2048_2048_3.h5')
model9 = load_model('2048_2048_4.h5')


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
    
class myagent1(Agent):
    def __init__(self, game,display=None):
        super().__init__(game, display)
        
    def step(self):
        x_to_pred = np.array(self.game.board)
        a=x_to_pred.reshape(16)
        b=np.sort(a)
        a1=b[-1]
        a2=b[-2]
        a3=b[-3]
        x_to_pred = np.log2(x_to_pred+1)
        x_to_pred = np.trunc(x_to_pred)
        if a1<128:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 7) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 7)
        
            pred=model1.predict(x_to_pred)  
        elif a1<256:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 8) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 8)
        
            pred=model2.predict(x_to_pred)  
        elif a1<512:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 9) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 9)
        
            pred=model3.predict(x_to_pred)  
        elif a1<1024 and a2<256:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 10) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 10)
        
            pred=model4.predict(x_to_pred)  
        elif a1<1024:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 10) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 10)
        
            pred=model5.predict(x_to_pred)  
        elif a1<2048 and a2<256:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 11) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 11)
        
            pred=model6.predict(x_to_pred)  
        elif a1<2048 and a2<512:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 11) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 11)
        
            pred=model7.predict(x_to_pred)  
        elif a1<2048 and a3<256:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 11) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 11)
        
            pred=model8.predict(x_to_pred)  
        elif a1<2048:
            x_to_pred = keras.utils.to_categorical(x_to_pred, 11) 
            x_to_pred = x_to_pred.reshape(1, 4, 4, 11)
        
            pred=model9.predict(x_to_pred)  
        r=pred[0]
        r1=r.tolist()
        direction=r1.index(max(r1))
        return direction