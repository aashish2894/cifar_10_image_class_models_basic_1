import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
	# Load the raw CIFAR-10 data
	cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
	# Subsample the data
	mask = list(range(num_training, num_training + num_validation))
	X_val = X_train[mask]
	y_val = y_train[mask]
	mask = list(range(num_training))
	X_train = X_train[mask]
	y_train = y_train[mask]
	mask = list(range(num_test))
	X_test = X_test[mask]
	y_test = y_test[mask]
	return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()



from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])



from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.


for lr in learning_rates:
	for r in regularization_strengths:
		print(lr)
		print(r)
		svm = LinearSVM()
		loss_hist = svm.train(X_train_feats, y_train, learning_rate=lr, reg=r, num_iters=1000, verbose=True)
		y_train_pred = svm.predict(X_train_feats)
		training_accuracy = np.mean(y_train == y_train_pred)
		y_val_pred = svm.predict(X_val_feats)
		validation_accuracy = np.mean(y_val == y_val_pred)
		results[(lr,r)] = (training_accuracy,validation_accuracy)
		if(validation_accuracy>best_val):
			best_val = validation_accuracy
			best_svm = svm
        
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr,reg in sorted(results):
    train_accuracy,val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    


print("best validation accuracy achieved during cross-validation: %f" % best_val)

# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)


# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
	idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
	idxs = np.random.choice(idxs, examples_per_class, replace=False)
	for i, idx in enumerate(idxs):
		plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
		plt.imshow(X_test[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(cls_name)

plt.show()


print(X_train_feats.shape)

from cs231n.classifiers.neural_net import TwoLayerNet




best_net = None
best_val = -1
best_stats = None
#learning_rates = [1e-4]
#regularization_strengths = [0.25]
learning_rates = np.logspace(-10, 0, 5) # np.logspace(-10, 10, 8) #-10, -9, -8, -7, -6, -5, -4
regularization_strengths = np.logspace(-3, 5, 5) 
results = {} 
iters = 2000 #100

input_size = X_train_feats.shape[1]
hidden_size = 500
num_classes = 10

#net = TwoLayerNet(input_size, hidden_size, num_classes)



for lr in learning_rates:
	for rs in regularization_strengths:
		print(lr)
		print(rs)
		net = TwoLayerNet(input_size, hidden_size, num_classes)
		# Train the network
		stats = net.train(X_train_feats, y_train, X_val_feats, y_val,num_iters=iters, batch_size=200,learning_rate=lr, learning_rate_decay=0.95,reg=rs, verbose=True)
		y_train_pred = net.predict(X_train_feats)
		acc_train = np.mean(y_train == y_train_pred)
		y_val_pred = net.predict(X_val_feats)
		acc_val = np.mean(y_val == y_val_pred)
		results[(lr, rs)] = (acc_train, acc_val)
		if best_val < acc_val:
			best_stats = stats
			best_val = acc_val
			best_net = net

# Print out results.

for lr, reg in sorted(results):
	train_accuracy, val_accuracy = results[(lr, reg)]
	print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)    


print 'best validation accuracy achieved during cross-validation: %f' % best_val 

plt.subplot(2, 1, 1)
plt.plot(best_stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(best_stats['train_acc_history'], label='train')
plt.plot(best_stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print(test_acc)