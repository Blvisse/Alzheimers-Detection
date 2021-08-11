"""
This script builds a Convoution Neural Network to help classify the images
"""


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,InputLayer,BatchNormalization,Dropout

class Network:

    def build_network(self,train,validation):

        model = Sequential()
        model.add(InputLayer(input_shape=(222,224,4)))

        #first layer
        model.add(Conv2D(25,(5,5),activation='relu',strides=(1,1),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

        # second convolution layers
        model.add(Conv2D(50,(5,5),activation='relu',strides=(2,2),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(BatchNormalization())
        #third layer
        model.add(Conv2D(60,(5,5),activation='relu',strides=(2,2),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(BatchNormalization())
        #fourth layer
        model.add(Conv2D(70,(3,3),activation='relu',strides=(2,2),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
        model.add(BatchNormalization())
        #flatten the output  
        model.add(Dense(units=100,activation='relu'))
        model.add(Dense(units=100,activation='relu')) 
        model.add(Dense(units=100,activation='relu'))
        model.add(Dense(units=100,activation='relu'))
        model.add(Dropout(0.25))

        # genrate the output layer 
        model.add(Dense(units=100,activation='softmax'))

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        model.fit_generator(train,epochs=30,validation_data=validation)

        return model.summary()



