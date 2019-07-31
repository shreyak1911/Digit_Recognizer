# Digit_Recognizer
### Data Description
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 

The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if it is predicted that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then the submission file would look like:

ImageId,Label
1,3
2,7
3,8 
(27997 more lines)

The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that have correctly classified all but 3% of the images.

### Implementation
-  The training of the system is done with the help of Keras. This involves a Convolutiuonal Neural Network which consists of 3 layers. The model used for the training is the Sequential Model. The training is done with the help of train.csv file which consists of 784 pixels along with the label of the decoding of the pixels. This training can be found in dr_train.py file.
Once the model is trained then it is saved in the form of a json file along with its weights in the form of a .h5 file.
-  After training, comes the testing of the model. For the testing the given file is test.csv. This file consists of 784 pixels without a description label. During the testing it is very important to recall the weights and the model created in the dr_train.py file. On processing of the test.csv, the predictions made are again saved into a separate csv file for evaluation. All the code for the testing can be found in dr_test.py file.
-  For checking the correctness of the predictions, the given pixels in the datasets can be decoded into greyscale images of the handwritten digits. All the code for this step can found in train_dec.py file. But in this file the decoded images of the pixels are saved in folders according to the labels attached with them.

### How to run
-  Save all the files in the same folder
-  Run the training file by giving the following command:  
'''bash
python3 dr_train.py
'''
-  After the training is complete, run the testing file:
'''bash
python3 dr_test.py
'''
-  The model, weights and the solution file will be saved in the same folder.
