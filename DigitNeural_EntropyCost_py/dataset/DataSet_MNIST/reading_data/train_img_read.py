#	Author: Dhruvin Modi
#	Email-Id: dhruvinmodi2015@gmail.com
#	Aim: Generate .csv files from Mnist Dataset and read .csv files generate a list that will be fed to the neural netork
#	WorkFlow:
#		1)Read train-images-idx3-ubyte/data (training images given by MNIST)
#		2)Read train-labels-idx1-ubyte/data (training lables given by MNIST)
#		3)Generate train_images.csv from bynary file (train-images-idx3-ubyte/data)
#			-train_images.csv Contains 60000 rows and 784 columns
#			-each row represens a single image
#		4)Generate train_lable.csv from bynary file (train-labels-idx1-ubyte/data)
#			-train_lable.csv Contains 60000 rows and 1 column
#			-Each digit represent a lable of each image
#		5)Generate a list of tuples of the form below
#			dataset = [(x0,y0), (x1,y1),.....(x60000,y60000)]
#					where x is a list of an image pixel values
#					where y is an integer value tha represents the lable of x 


import struct
images = open("../train-images-idx3-ubyte/data","rb")
lable = open("../train-labels-idx1-ubyte/data","rb")
im_read = images.read()
lb_read = lable.read()
images_write = open("train_images.csv","w")
lable_write = open("train_lable.csv","w")
temp = 0
j = 1
k = 0
print("Working...")
for i in im_read:
	if temp > 15:
		i = int(i)
		images_write.write('%d,'%i)
	if temp > 15:
		if j%784 == 0:
			images_write.write('\n')
			j = 0
			k += 1
		j += 1
	temp += 1
temp = 0
for i in lb_read:
	if temp > 7:
		i = int(i)
		lable_write.write('%d,\n'%i)
	temp += 1
images.close()
lable.close()
images_write.close()
lable_write.close()
images = open("train_images.csv","r")
lable = open("train_lable.csv","r")
dataset = []
for y,x in zip(lable, images):
	y = y.split(',')
	x = x.split(',')
	del y[1]
	y = int(y[0])
	del x[784]
	for i in range(0,784):
		x[i] = int(x[i])
	dataset.append((x,y))
print("done")
