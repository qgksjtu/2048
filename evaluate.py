from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent,myagent1
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


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


   


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=myagent1)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
