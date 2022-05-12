# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
import csv 


nb_reviews = int(sys.argv[2])
text_output = sys.argv[3]


def sample_prediction(prediction):
    X = prediction[0]
    rnd_idx = numpy.random.choice(len(X), p=X)
    return rnd_idx


sentiment_type = int(sys.argv[4])

# load ascii text and covert to lowercase
df = pd.read_csv("ReviewsLabelled.csv", names=['sentence','sentiment', 'source'], header=0,sep='\t', encoding='utf8')
df_sentiment = df.loc[df['sentiment'] == sentiment_type]
df_reviews = df_sentiment['sentence']

separator_char = '|'

# load ascii text and covert to lowercase
raw_text = df_reviews.str.cat(sep=separator_char)
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
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
# load the network weights
filename = sys.argv[1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
#print("Seed:")
#print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

output_file = open(text_output, 'w')

# Uncomment to write the seed at the beginning of the output
#output_file.write(''.join([int_to_char[value] for value in pattern]))

max_review_length = 100

#output_file.write("sentence\tsentiment\tsource\n")

# generate characters
for i in range(nb_reviews):
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	char_count = 0
	result = ''
	
	while (char_count < max_review_length and result != separator_char):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		#index = sample_prediction(prediction)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		if result != separator_char:	
			#sys.stdout.write(result)
			output_file.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
		char_count += 1	
	output_file.write('\t' + str(sentiment_type) + '\tnull\n')
	print("Generated %d reviews\r"%(i+1), end="")
print("\nDone.")

output_file.close()

