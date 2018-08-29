import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import LSTM, LSTMCell, StackedRNNCells, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn.utils import shuffle as shuff

num_epochs = 3
batch_size = 5
alpha = .01


class LossPlot(Callback):
    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        print("Loss: {} \n Recall: {}".format(logs.get('loss'), logs.get('recall')))


def process_data(fft=False, normalize=False, smote=False, shuffle=True):
    train = pd.read_csv('data/exoTrain.csv')
    test = pd.read_csv('data/exoTest.csv')

    y = (train['LABEL'].values + 1) % 2
    y_ = (test['LABEL'].values + 1) % 2

    X = train.loc[:, 'FLUX.1':].values
    X_ = test.loc[:, 'FLUX.1':].values

    X_fft = np.fft.fft(X)[..., np.newaxis].real
    X_fft_ = np.fft.fft(X_)[..., np.newaxis].real

    X, y = shuff(X_fft, y)
    X_, y_ = shuff(X_fft_, y_)
    return X, y, X_, y_


def recall(y_true, y_pred):
    return recall_score(y_true=K.eval(y_true), y_pred=K.eval(y_pred))


def create_model(batch_size, dropout=0.0, recurrent_state_dropout=0.0):
    model = Sequential()
    # model.add(LSTM(128,
    #                return_sequences=True,
    #                batch_input_shape=(batch_size, 3197, 1),
    #                dropout=dropout,
    #                recurrent_dropout=recurrent_state_dropout,
    #                stateful=True))

    model.add(LSTM(10,
                   return_sequences=True,
                   dropout=dropout,
                   batch_input_shape=(batch_size, 3197, 1),
                   recurrent_dropout=recurrent_state_dropout,
                   stateful=True))

    model.add(Flatten())

    model.add(Dense(1))

    ada = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    model.compile(loss='binary_crossentropy',
                  optimizer=ada,
                  metrics=[])
    return model


model = create_model(batch_size)

model.fit(x=X[:5085], y=y[:5085], epochs=num_epochs, batch_size=batch_size, callbacks=[LossPlot()])

score = model.evaluate(x=x_fft_, y=label_, batch_size=batch_size)

# train_dataset = tf.data.Dataset.from_tensor_slices((x_fft, label)).repeat().batch(30)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_fft_, label_)).repeat().batch(30)
#
# iter_ = train_dataset.make_one_shot_iterator()
# x, y = iter_.get_next()
#
#
# def build_lstm_layers(input, layer_sizes, keep_prob):
#     print("input shape: {}".format(input.shape))
#     lstms = [tf.contrib.rnn.BasicLSTMCell(size, dtype=tf.float64) for size in layer_sizes]
#     drop_wrappers = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for lstm in lstms]
#
#     cell = tf.contrib.rnn.MultiRNNCell(drop_wrappers)
#
#     init_state = cell.zero_state(batch_size, dtype=tf.float64)
#     outputs, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, dtype=tf.float64)
#     return init_state, outputs, cell, final_state
#
#
# def build_plo(learning_rate):
#     print("output shape: {}".format(outputs[-1].shape))
#     preds = tf.contrib.layers.fully_connected(outputs[:, :, -1], 2, activation_fn=tf.sigmoid)
#     print(preds)
#     loss = tf.losses.mean_squared_error(preds, y)
#     optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
#     return preds, loss, optimizer
#
#
# initial_state, outputs, cell, final_state = build_lstm_layers(input=x, layer_sizes=[512, 256], keep_prob=.5)
# preds, loss, optimizer = build_plo(learning_rate=alpha)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(num_epochs):
#         print("Running epoch {}".format(i + 1))
#         state = sess.run(initial_state)
#         sess.run(optimizer)
#         print("Loss for this epoch: {}".format(sess.run(loss)))
