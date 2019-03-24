"""
Name: Dhruvin Modi
E-Mail: dhruvinmodi2015@gmail.com
"""
import random
import numpy as np
import matplotlib.pyplot as plt
class network(object):
	def __init__(self,sizes):
		self.sizes = sizes
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		self.biases = [np.random.randn(x,1) for x in sizes[1:]]
		self.dataset = []
		self.test_dataset = []
		self.percentageList = []
		self.same = 0
		self.diff = 0
		self.test = 0
	def read_dataset(self):
		f = open('./dataset/Iris.csv','r')
		for line in f:
			line = line.replace(line[-1],"")
			Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = line.split(",")
			SepalLengthCm = float(SepalLengthCm) / 10.0
			SepalWidthCm = float(SepalWidthCm) / 10.0
			PetalLengthCm = float(PetalLengthCm) / 10.0
			PetalWidthCm = float(PetalWidthCm)
			if Species == 'Iris-setosa':
				Species = 0
			elif Species == 'Iris-versicolor':
				Species = 1
			elif Species == 'Iris-virginica':
				Species = 2
			Species = int(Species)
			x = np.asarray([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm])
			y = np.zeros((3,1))
			y[Species] = 1
			self.dataset.append(np.asarray([x,y]))
	def read_test_dataset(self):
		f = open('./dataset/test.csv','r')
		for line in f:
			line = line.replace(line[-1],"")
			Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = line.split(",")
			SepalLengthCm = float(SepalLengthCm) / 10.0
			SepalWidthCm = float(SepalWidthCm) / 10.0
			PetalLengthCm = float(PetalLengthCm) / 10.0
			PetalWidthCm = float(PetalWidthCm)
			if Species == 'Iris-setosa':
				Species = 0
			elif Species == 'Iris-versicolor':
				Species = 1
			elif Species == 'Iris-virginica':
				Species = 2
			Species = int(Species)
			x = np.asarray([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm])
			y = np.zeros((3,1))
			y[Species] = 1
			self.test_dataset.append(np.asarray([x,y]))
		return self.test_dataset
	def sigmoid(self,z):
		return(1.0 / (1.0 + np.exp(-z)))
	def derv_sigmoid(self,z):
		return (self.sigmoid(z)*(1-self.sigmoid(z)))
	def train(self, epochs, test_dataset):
		for epo in range(0,epochs):
			data_lim = 1
			print("Epochs: ",epo + 1)
			index = 0
			avg_nebla_w = [np.zeros((y,x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
			avg_nebla_b = [np.zeros((x,1)) for x in self.sizes[1:]]
			random.shuffle(self.dataset)
			for i in self.dataset:
				activations = []
				z = []
				index = 0
				i[0] = i[0].reshape(self.sizes[0],1)
				activations.append(i[0])
				for w,b in zip(self.weights,self.biases):
					weighted_sum = np.dot(w,activations[index]) + b
					z.append(weighted_sum)
					a = self.sigmoid(weighted_sum)
					activations.append(a)
					index += 1
					if index == len(self.sizes) - 1:
						index = 0
						nebla_w,nebla_b = self.backpropogation(activations,z,i[1])
						avg_nebla_w = [avg_neb_w + neb_w for avg_neb_w,neb_w in zip(avg_nebla_w,nebla_w[::-1])]
						avg_nebla_b = [avg_neb_b + neb_b for avg_neb_b,neb_b in zip(avg_nebla_b,nebla_b[::-1])]
				data_lim += 1
				if data_lim == 90:
					self.weights = [weights - ((avg_w * 0.5) / 90) for weights,avg_w in zip(self.weights,avg_nebla_w)]
					self.biases = [biases - ((0.5*avg_b) / 90) for biases,avg_b in zip(self.biases,avg_nebla_b)]
					avg_nebla_w = [np.zeros((y,x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
					avg_nebla_b = [np.zeros((x,1)) for x in self.sizes[1:]]
					data_lim = 0
			self.feedforword(test_dataset)
			self.performance()
			self.same = 0
			self.diff = 0

	def backpropogation(self,activations,z,y):
		nebla_b = []
		nebla_w = []
		layers_index = len(self.sizes)
		delta = 2 * self.derv_sigmoid(z[-1]) * self.cost(activations[-1],y)
		weights = delta * np.transpose(activations[-2])
		nebla_b.append(delta)
		nebla_w.append(weights)
		layers_index -= 1 #3
		while layers_index > 1:
			delta = self.derv_sigmoid(z[layers_index - 2]) * np.dot((self.weights[layers_index - 1]).transpose() , delta)
			nebla_b.append(delta)
			weights = delta * np.transpose(activations[layers_index - 2])
			nebla_w.append(weights)
			layers_index -= 1
		return nebla_w,nebla_b
	def cost(self,last_activation,y):
		return (last_activation - y)
	def feedforword(self,dataset):
		for i in dataset:
			activations = []
			i[0] = i[0].reshape(len(self.sizes),1)
			activations.append(i[0])
			index = 0
			sum_cost = 0
			for w,b in zip(self.weights,self.biases):
				weighted_sum = np.dot(w ,activations[index]) + b
				a = self.sigmoid(weighted_sum)
				activations.append(a)
				index += 1
			self.evaluate(activations[len(self.sizes) - 1],i[1])
	def evaluate(self, obtain, actual):
		index = np.argmax(obtain)
		self.test += 1
		if actual[index] == 0:
			self.diff += 1
		else:	self.same += 1
	def performance(self):
		print("Sucess : ", self.same,"/",len(self.test_dataset))
		print("Fail : ", self.diff,"/",len(self.test_dataset))
		print("Accuracy: ",(100 * self.same) / len(self.test_dataset))
		self.percentageList.append((100 * self.same) / len(self.test_dataset))
	def plot_percentage(self):
		plt.plot(self.percentageList)
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.title('Accuracy after each epoch')
		plt.savefig('./accuracy_after_each_epoch.png')
		plt.close()

obj = network(np.asarray([4,5,3,3]))
obj.read_dataset()
test_dataset = obj.read_test_dataset()
obj.train(1000, test_dataset)
obj.plot_percentage()
