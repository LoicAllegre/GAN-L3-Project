from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot

def generate_latent_points(latent_dim, n_samples, n_classes=10):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def save_plot(examples, n_classes, n_per_class):
	pyplot.figure()
	for i in range(n_classes):
		for j in range(n_per_class):
			pyplot.axis('off')
			pyplot.imshow(examples[i*n_per_class + j, :, :, 0], cmap='gray')
			pyplot.savefig('generated/{:01d}/{:05d}.png'.format(i, j), bbox_inches='tight', pad_inches = 0)
			pyplot.clf()
			print("Generated{:05d} images\r".format(i*n_per_class+j))


model = load_model('cgan_generator.h5')
latent_points, labels = generate_latent_points(100, 50000)
labels = asarray([x for x in range(10) for _ in range(5000)])
X  = model.predict([latent_points, labels])
X = (X + 1) / 2.0
save_plot(X, 10, 5000)