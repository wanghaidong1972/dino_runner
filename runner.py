# coding: utf-8

from dinoenv import DinoEnv
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import load_model
import random
import pickle
import os
import time
import hickle

BUFFER_SIZE = 48000

ACTIONS = 2  #  jump or keep stay
GAMMA = 0.99  # decay rate
BATCH_SIZE = 16  # size of minibatch
LEARNING_RATE = 1e-4
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # minimum epsilon to keep exploration
INITIAL_EPSILON = 0.1

BUFFER_FILE = "buffer.hd5"
MODEL_FILE = "current.model"
PARA_FILE = "parameters.pkl"

img_rows, img_cols = 80, 80 #image size
img_channels = 4  # We stack 4 frames


def buildmodel(model_file = None):
    if model_file:
        model = load_model(model_file)
        print("load model from existing file")
    else:
        print("build the model from beginning")
        input_shape = (img_rows, img_cols, 4)
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(4, 4),
                   padding='same',
                   strides=(2, 2),
                   activation='relu',
                   input_shape=input_shape))  # 80*80*4
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(ACTIONS))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

    return model


def save_obj(obj, name):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name , 'rb') as f:
        return pickle.load(f)

def save_hickle(obj, name):
    hickle.dump(obj, name, mode='w')


def load_hickle(name):
    return hickle.load(name)

def save_breakpoint(parameters,model,data_buffer):
    model.save(MODEL_FILE)
    save_obj(parameters,PARA_FILE)
    # save_obj(data_buffer, BUFFER_FILE)
    save_hickle(data_buffer, BUFFER_FILE)

def train(env):
    mode_file = MODEL_FILE if os.path.exists(MODEL_FILE) else None
    model = buildmodel(mode_file)
    current_s = env.get_current_status()  # get the first stack of images
    if os.path.exists(PARA_FILE):
        epsilon, step = load_obj(PARA_FILE)
    else:
        epsilon = INITIAL_EPSILON
        step = 0

    last_time = time.time()

    while True:
        Q_value = 0
        step += 1

        if epsilon > FINAL_EPSILON and step > OBSERVATION:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if random.random() <= epsilon:  # exploration
            print("--------------random action ----------")
            action = random.randrange(ACTIONS)
        else:  # exploitation
            current_s = current_s.reshape(1,img_cols, img_rows,img_channels)
            q = model.predict(current_s)  # get the prediction by current status
            max_Q = np.argmax(q)  # choose an action with maximum q value
            action = max_Q

        next_s, reward, isover, others = env.step(action) # step forward and get what happened
        data_buffer.append((current_s, action, reward, next_s, isover)) # save those to buffer

        if isover:
            current_s = env.get_ini_status()
        else :
            current_s = next_s

        if (step) % 10000 == 0:
            env.pause()
            save_breakpoint((epsilon,step),model,data_buffer)
            env.resume()

        if len(data_buffer) > OBSERVATION:
            # begin training
            minibatch = random.sample(data_buffer, BATCH_SIZE)
            status_0 = minibatch[0][0]
            inputs = np.zeros((BATCH_SIZE, status_0.shape[1], status_0.shape[2], status_0.shape[3]))  # 16, ,80, 80,4
            targets = np.zeros((BATCH_SIZE, ACTIONS))  # 16, 2

            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]  # stack of images
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]  # reward at state_t and action_t
                state_t1 = minibatch[i][3]  # next state
                gameover = minibatch[i][4]  # if game over

                inputs[i:i + 1] = state_t  # use i:i+1 to keep shape as (1,80,80,4) for debug
                # inputs[i] = state_t
                targets[i] = model.predict(state_t)  # predicted q values from current step
                Q_value = model.predict(state_t1)  # predict q values for next step

                if gameover:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_value)  # with future reward
            # env.pause()
            loss = model.train_on_batch(inputs, targets)
            # env.resume()
            if step%50 == 0:
                print ("step is {},epsilon is {},loss is {} ,Q_value is {}".format(step,epsilon,loss,Q_value))
                print('fps: {0}'.format(1 / (time.time() - last_time)))  #  frame rate

        last_time = time.time()

env = DinoEnv("chromedriver.exe") if os.name=="nt" else DinoEnv("./chromedriver")
# data_buffer = load_obj(BUFFER_FILE) if os.path.exists(BUFFER_FILE) else deque(maxlen=BUFFER_SIZE)
data_buffer = load_hickle(BUFFER_FILE) if os.path.exists(BUFFER_FILE) else deque(maxlen=BUFFER_SIZE)

try:
    train(env)
except Exception as e:
    env.close()
    raise e


