import numpy as np
import csv
from PIL import Image
#from time import sleep


# Decoding the pixels given in the csv file to images
#and saving them in their respective folders according to their labels
i = 0
with open('test.csv') as file:
	reader = csv.reader(file)
	header = next(reader)
	for row in reader:
		
			# label = np.int32(row[0])
			d = np.array(row[0:], dtype=np.uint8).reshape(28, 28)	
			d_img = Image.fromarray(d, 'P')
			d_img.save('zen'+str(n)+'.png')
			i+=1 

			if i>20:
				break
