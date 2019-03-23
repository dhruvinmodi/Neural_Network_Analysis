"""
Name: Dhruvin Modi
E-Mail: dhruvinmodi2015@gmail.com
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.uniform(-1,1,(y,x)) for x,y in zip(sizes[ : -1],sizes[1 : ])]
		self.percentageList = []
		self.imgCounter = 0
	def dataset(self):
		images = open("./dataset/DataSet_MNIST/reading_data/train_images.csv","r")
		lable = open("./dataset/DataSet_MNIST/reading_data/train_lable.csv","r")
		dataset = []
		for y,x in zip(lable, images):
			y = y.split(',')
			x = x.split(',')
			del y[1]
			y = int(y[0])
			del x[784]
			for i in range(0,784):
				x[i] = int(x[i])
				x[i] /= 255
			x = np.array(x)
			x = x.reshape((784,1))
			dataset.append([x,y])
		return dataset
	def feedforward(self,a):
		a = a.reshape(self.sizes[0],1)
		for b,w in zip(self.biases,self.weights):
			a=self.sigmoid(np.dot(w, a) + b)
		i,j = np.where(a == np.amax(a))
		return(i)
	def sigmoid(self,z):
		"""Sigmoid function"""
		return(1.0 / (1.0 + np.exp(-z)))
	def derv_sigmoid(self,z):
		"""Derivative of the sigmoid function."""
		return (self.sigmoid(z)*(1-self.sigmoid(z)))
	def train(self, training_data, epochs, eta, testing_data_x, testing_data_y):
		max_per = 0
		index = 0
		for k in range(0,epochs):
			img_no = 0
			print()
			print("Epochs: ",(k + 1) )
			avg_nebla_b = [np.zeros((p, 1))for p in self.sizes[1:]]
			avg_nebla_w = [np.zeros((q, p))for p,q in zip(self.sizes[ : -1],self.sizes[1 : ])]
			random.shuffle(training_data)
			for i in training_data:
				img_no += 1
				index = 0
				activations = []
				z = []
				nebla_b = []
				nebla_w = []
				y = np.zeros((10,1))
				y[i[1]] = 1
				activations.append(i[0])
				for b,w in zip(self.biases,self.weights):
					weighted_sum = np.dot(w,activations[index]) + b
					z.append(weighted_sum)
					a=self.sigmoid(weighted_sum)
					activations.append(a)
					index += 1
					if index == (len(self.sizes) - 1):
						index = 0
						nebla_b, nebla_w = self.backprop(activations,z,y)
						nebla_b = [ p for p in nebla_b[::-1]]
						nebla_w = [ p for p in nebla_w[::-1]]
						avg_nebla_b = [avg + nb for avg,nb in zip(avg_nebla_b,nebla_b)]
						avg_nebla_w = [avg + nw for avg,nw in zip(avg_nebla_w,nebla_w)]
				if img_no % 10 == 0:
					self.weights = [w - (eta/10)* nw for w,nw in zip(self.weights,avg_nebla_w)]
					self.biases = [b - (eta/10) * nb for b,nb in zip(self.biases,avg_nebla_b)]
					avg_nebla_b = [np.zeros((p, 1))for p in self.sizes[1:]]
					avg_nebla_w = [np.zeros((q, p))for p,q in zip(self.sizes[ : -1],self.sizes[1 : ])]
			self.imgCounter += 1
			self.multi_plot_histogram()
			per = self.percentage(testing_data_x, testing_data_y)
			if per > max_per:
				self.storage()
				max_per = per
	def backprop(self,activations,z,y):
		nebla_b = []
		nebla_w = []
		delta = (self.sigmoid(z[len(self.sizes) - 2]) - y)
		nebla_b.append(delta) 
		nebla_w.append(np.dot(delta,activations[len(self.sizes) - 2].transpose()))
		for l in range(len(self.sizes) - 1,1,-1):
			delta = (self.derv_sigmoid(z[l - 2]) * np.dot(self.weights[l - 1].transpose(),delta))
			nebla_b.append(delta)
			nebla_w.append(np.dot(delta, activations[l - 2].transpose()))
		return (nebla_b,nebla_w)
	def storage(self):
		biases_file = open("./best_weights_biases/biases.txt", "w")
		weights_file = open("./best_weights_biases/weights.txt", "w")
		for b,w in zip(self.biases,self.weights):
			for i in b:
				biases_file.write("%d, " % i)
			biases_file.write("\n")
			for i in w:
				for j in i:
					weights_file.write("%d, "%j)
				weights_file.write("\n")
			weights_file.write("\n")
		biases_file.close()
		weights_file.close()
		biases_file = open("./best_weights_biases/biases.csv", "w")
		weights_file = open("./best_weights_biases/weights.csv", "w")
		for b,w in zip(self.biases,self.weights):
			for i in b:
				biases_file.write("%d," % i)
			biases_file.write("\n")
			for i in w:
				for j in i:
					weights_file.write("%d,"%j)
				weights_file.write("\n")
			weights_file.write("\n")
		biases_file.close()
		weights_file.close()
	def test(self):
		images = open("./dataset/DataSet_MNIST/reading_data/test_images.csv","r")
		lable = open("./dataset/DataSet_MNIST/reading_data/test_lable.csv","r")
		X = []
		Y = []
		for y,x in zip(lable, images):
			y = y.split(',')
			x = x.split(',')
			del y[1]
			y = int(y[0])
			del x[784]
			for i in range(0,784):
				x[i] = int(x[i])
				x[i] /= 255
			x = np.array(x)
			x = x.reshape((784,1))
			X.append(x)
			Y.append(y)
		return (X,Y)
	def percentage(self, testing_data_x, testing_data_y):
		same = 0
		diff = 0
		for x,y in zip(testing_data_x, testing_data_y):
			a = net.feedforward(x)
			if len(a) > 1 :
				diff += 1
				continue
			if a == y:
				same += 1
			else:
				diff += 1
		self.percentageList.append((100*same) / 10000)
		print("Correct Detected Images: ", same, "/10000")
		print("Wrong Detected Images: ", diff, "/10000")
		print("Accuracy in Percentage: ", (100*same) / 10000)
		return ((100*same) / 10000)
	def plot_percentage(self):
		plt.plot(self.percentageList)
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.title('Accuracy after each epoch')
		plt.savefig('./graphs/accuracy_after_each_epoch.png')
		plt.close()
	def plot_histogram_before_learning(self):
		weights = []
		for i in self.weights:
			for j in i:
				for k in j:
					weights.append(k)
		plt.hist(weights)
		plt.xlabel('Weights')
		plt.ylabel('Frequecy')
		plt.title('Weight histogram before learning')
		plt.savefig('./graphs/weights_before_learning_histogram.png')
		plt.close()
		biases = []
		for i in self.biases:
			for j in i:
				biases.append(j)
		plt.hist(biases)
		plt.xlabel('Biases')
		plt.ylabel('Frequecy')
		plt.title('Bias histogram before learning')
		plt.savefig('./graphs/biases_before_learning_histogram.png')
		plt.close()
	def plot_histogram(self):
		weights = []
		for i in self.weights:
			for j in i:
				for k in j:
					weights.append(k)
		plt.hist(weights)
		plt.xlabel('Weights')
		plt.ylabel('Frequecy')
		plt.title('Weight histogram after learning')
		plt.savefig('./graphs/weights_after_learning_histogram.png')
		plt.close()
		biases = []
		for i in self.biases:
			for j in i:
				biases.append(j)
		plt.hist(biases)
		plt.xlabel('Biases')
		plt.ylabel('Frequecy')
		plt.title('Bias histogram after learning')
		plt.savefig('./graphs/biases_after_learning_histogram.png')
		plt.close()
	def multi_plot_histogram(self):
		weights = []
		for i in self.weights:
			for j in i:
				for k in j:
					weights.append(k)
		plt.hist(weights)
		plt.xlabel('Weights')
		plt.ylabel('Frequecy')
		plt.title('Epoch: ' + str(self.imgCounter))
		temp = "./histogram_after_each_epoch/weights/" + str(self.imgCounter) + ".png"
		plt.savefig(temp)
		plt.close()
		biases = []
		for i in self.biases:
			for j in i:
				biases.append(j)
		plt.hist(biases)
		plt.xlabel('Biases')
		plt.ylabel('Frequecy')
		plt.title('Epoch: ' + str(self.imgCounter))
		temp = "./histogram_after_each_epoch/biases/" + str(self.imgCounter) + ".png"
		plt.savefig(temp)
		plt.close()
	def videoMaker(self):
		W_images = []
		for i in range(1,self.imgCounter + 1):
			W_images.append(str(i) + '.png')
		W_images.append(str(self.imgCounter) + '.png')
		frame = cv2.imread(os.path.join('./histogram_after_each_epoch/weights', W_images[0]))
		height, width, layers = frame.shape
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		video = cv2.VideoWriter('./video_of_histogram/weights.mp4', fourcc, 1, (width,height))
		for image in W_images:
		    video.write(cv2.imread(os.path.join('./histogram_after_each_epoch/weights', image)))
		cv2.destroyAllWindows()
		video.release()
		B_images = []
		for i in range(1,self.imgCounter + 1):
			B_images.append(str(i) + '.png')
		B_images.append(str(self.imgCounter) + '.png')
		frame = cv2.imread(os.path.join('./histogram_after_each_epoch/biases', B_images[0]))
		height, width, layers = frame.shape
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		video = cv2.VideoWriter('./video_of_histogram/biases.mp4', fourcc, 1, (width,height))
		for image in B_images:
		    video.write(cv2.imread(os.path.join('./histogram_after_each_epoch/biases/', image)))
		cv2.destroyAllWindows()
		video.release()

epochs = 60
eta = 2
nu_struture = [784,60,30,10]
net = Network(nu_struture)
net.plot_histogram_before_learning()
print("Dataset Reading Started.....")
training_data = net.dataset()
print("Testing data reading....")
testing_data_x,testing_data_y = net.test()
print("Training Started.....")
net.train(training_data,epochs,eta,testing_data_x, testing_data_y)
net.plot_percentage()
net.plot_histogram()
net.videoMaker()
