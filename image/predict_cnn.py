# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
import numpy as np

def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def predict_image(image_path):
	img = load_image(image_path)
	model = load_model('final_model.h5')
	predict_x=model.predict(img)
	classes_x=np.argmax(predict_x,axis=1)
	print(classes_x[0])

predict_image(sys.argv[1])