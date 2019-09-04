# Convolutional Neural Network



# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D # used for first step, for convolutional layers
from keras.layers import MaxPooling2D # used for pooling step, pooling layers
from keras.layers import Flatten # used for flattening
from keras.layers import Dense #used for Dense

# Initialising CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # relu for non-linearity


# Pooling
#max pooling : to reduce size of feature maps
# keeping the important parts of pictures while reducing size

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection

classifier.add(Dense(units = 128, activation = 'relu')) #common practice to pick a power of 2
classifier.add(Dense(units = 6, activation = 'sigmoid'))

# Compiling CNN

# stochastic algorithm 'adam'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,  # scale turns to 0~1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) #change scale to 0~1

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32
                                                 )

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32
                                            )

classifier.fit_generator(training_set,
                         steps_per_epoch = 42, #1339/32
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 5) # or 165/32

#Testing individual data
import numpy as np
from keras.preprocessing import image


# 파일이름 => 테스트하고싶은 이미지 파일명
test_image = image.load_img('validation/파일이름.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


result = classifier.predict(test_image)
training_set.class_indices


if result[0][0] == 1:
    prediction = 'apple'
elif result[0][1] == 1:
    prediction = 'beer'
elif result[0][2] == 1:
    prediction = 'egg'
elif result[0][3] == 1:
    prediction = 'mandarin'
elif result[0][4] == 1:
    prediction = 'milk'
elif result[0][5] == 1:
    prediction = 'soju'
else:
    prediction = 'error'

print (prediction)

print (prediction)
