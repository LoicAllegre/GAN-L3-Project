from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt



# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


def calculate_scores(y_test,y_pred):
    labels = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]

    print("Accuracy : %0.3f"%(accuracy_score(y_test,y_pred)))
    print("Classification Report")
    print(classification_report(y_test,y_pred,digits=5))    
    cnf_matrix = confusion_matrix(y_test,y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)
    plt.show()




def run_test(n_classes, samples_per_class):
    model = load_model('final_model.h5')
    y_pred = []
    y_test = []
    for clothes_class in range(n_classes):
        for image in range(samples_per_class):
            img_path = 'generated/{:01d}_{:03d}.png'.format(clothes_class, image)
            img = load_image(img_path)
            prediction = model.predict(img)
            classes = np.argmax(prediction,axis=1)[0]
            y_pred.append(classes)
            y_test.append(clothes_class)
    
    calculate_scores(y_test, y_pred)


run_test(10,10)