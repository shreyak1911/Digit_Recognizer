import numpy as np
import pandas as pd
from keras.models import model_from_json

def get_model():
    # defination to load the already trained model and the weighted model
    model = model_from_json(open('trained_model.json').read())
    model.load_weights('weighted_model.h5')
    return model

# retrieving the dataset to be tested
def get_csv_testdata():
    data = pd.read_csv('test.csv')
    imgs = data.values
    imgs = np.multiply(imgs,1.0/255.0)
    imgs = imgs.reshape(imgs.shape[0],1,28,28)
    return imgs

if __name__=='__main__':
    X_test = get_csv_testdata()
    model = get_model()
    y_test = np.argmax(model.predict(X_test,batch_size=50,verbose=1),axis=1)

    # Saving the tested data in a separate csv file:
    np.savetxt('solution_test.csv',np.c_[range(1,X_test.shape[0]+1),y_test],delimiter=',',header = 'Image_Id,Predicted_Label',comments = '',fmt='%d')
    print('Solutions of the test data File Generated!')
