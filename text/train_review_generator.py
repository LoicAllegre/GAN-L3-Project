import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
import sys
from tensorflow.keras.utils import plot_model

sentiment_type = int(sys.argv[1])

df = pd.read_csv("ReviewsLabelled.csv", names=['sentence','sentiment', 'source'], header=0,sep='\t', encoding='utf8')
df_sentiment = df.loc[df['sentiment'] == sentiment_type]
df_reviews = df_sentiment['sentence']

separator_char = '|'

# load ascii text and covert to lowercase
raw_text = df_reviews.str.cat(sep=separator_char)
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = ("models/positive-{epoch:02d}-{loss:.4f}.hdf5" if sentiment_type == 1 else "models/negative-{epoch:02d}-{loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
#model.fit(X, y, epochs=100, batch_size=64, callbacks=callbacks_list)

plot_model(model, to_file='figs/lstm_model_plot.png', show_shapes=True, show_layer_names=True)