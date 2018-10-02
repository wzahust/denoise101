# import h5py
# import matplotlib.pyplot as plt
import numpy as np
import keras
import scipy.io
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization


data = scipy.io.loadmat('Denoise_train.mat')
x = data['x']
y = data['y']
x = x.astype('float32')
y = y.astype('float32')
y = y.reshape(y.shape[0],64,64,1)
x = x.reshape(x.shape[0],64,64,1)
# y=x-y
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03,random_state=0)
# print('x shape',x.shape)

batch_size = 100
epochs = 100
# img_rows,img_cols = 64,64



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(64,64,1)))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, axis=1))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3, 3),padding='same'))
model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, axis=1))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3, 3),padding='same'))
model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, axis=1))
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding='same'))
model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, axis=1))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3),padding='same'))
model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, axis=1))
model.add(Activation('relu'))

model.add(Conv2D(1, kernel_size=(3, 3),padding='same'))

model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam())
model.fit( x_train ,y_train , batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

out = model.predict(x_test)
x_test = x_test.reshape(x_test.shape[0],64,64)
out = out.reshape(out.shape[0],64,64)
scipy.io.savemat('out.mat',{'in':x_test,'out':out})
# print(out.shape)

# from keras.utils import plot_model
# plot_model(model,show_shapes=True,to_file='model.png')



