import numpy as np
import csv
from PIL import Image
#from time import sleep
import os
# import pdb

newpath = '/home/shreyak/flaskProject/Digit_Recogniser'
# Decoding the pixels given in the train.csv file to images
# and saving them in their respective folders according to their labels
i = 0
with open('Data/train.csv') as file:
	read = csv.reader(file)
	head = next(read)
	for row in read:
		
			label = np.int32(row[0])
			print("LABEL", label, sep="\n")
			d = np.array(row[1:], dtype=np.uint8).reshape(28, 28)	
			d_img = Image.fromarray(d, 'P')
			
			# done for the first 50 images in the file
			if i == 10:
				break

			print("I:", i)
			img_name = 'zen_'+str(i)+'.png'

			# Making a new directory if it does not exists else saving the image on the pre-created directory
			if not os.path.exists(os.path.join(newpath, str(label))):
				os.makedirs(os.path.join(newpath, str(label)))
				
			print("SAVING THE DECODED IMAGE")
			d_img.save(os.path.join(os.path.join(newpath, str(label)), img_name))
			print("IMAGE SAVED SUCCESSFULLY")
			i += 1
			