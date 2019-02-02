# -*- coding: utf-8 -*-
"""

@author: Nomula Dheeraj Kumar

"""
#LOADING THE NECESSARY LIBRARIES
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import cv2
import numpy as np

#LOADING THE DATASET
data=mnist.load_data()
((X_train,y_train),(X_test,y_test))=data

#VISUALISING THE DATA
fig=plt.figure()
eg1=fig.add_subplot(1,2,1)
eg1.imshow(X_train[0],cmap='gray')
eg2=fig.add_subplot(1,2,2)
eg2.imshow(X_train[1],cmap='gray')

#RESHAPING THE DATASET
size_x=X_train.shape
X_train=X_train.reshape(size_x[0],size_x[1]*size_x[2])
size_x=X_test.shape
X_test=X_test.reshape(size_x[0],size_x[1]*size_x[2])
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#ENCODING THE CATEGORICAL DATA
onehotencoder=OneHotEncoder()
y_train=onehotencoder.fit_transform(y_train).toarray()
y_test=onehotencoder.transform(y_test).toarray()

#SCALING THE DATASET
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#BUILDING THE NETWORK
network=Sequential()
network.add(Dense(units=512,activation='sigmoid',input_dim=28*28,kernel_initializer='uniform'))
network.add(Dense(units=10,activation='softmax',kernel_initializer='uniform'))


network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

network.fit(X_train,y_train,batch_size=128,epochs=50)

y_pred=network.predict(X_test)

#EVALUATING THE MODEL
test_loss, test_acc = network.evaluate(X_test,y_test,verbose=0)
print(test_acc)

#SAVING THE MODEL
network.save('D:\Study\project\handwritten.h5')

#LOADING THE MODEL
classifier=load_model('D:\Study\project\handwritten.h5')

#RUNTIME DETECTION

video=cv2.VideoCapture(0)

while True:
    check,img=video.read()
    if check==True:
        cv2.rectangle(img,(0,0),(200,200),(255,0,0),2)
        cv2.putText(img,'Place a digit in the above box .. ',(0,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),3,cv2.LINE_AA)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray=gray[0:200,0:200]
        gray=cv2.resize(gray,(28,28))
        gray=gray.reshape(1,gray.shape[0],gray.shape[1],1)
        prediction=np.argmax(classifier.predict(gray))
        print(prediction)
        cv2.putText(img,'The Digit is : '+str(prediction),(0,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        cv2.imshow('Result',img)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
video.release()
cv2.destroyAllWindows()