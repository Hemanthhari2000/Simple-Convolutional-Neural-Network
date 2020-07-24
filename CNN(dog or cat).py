#==============================================================================  
#                         PART 1
# =============================================================================
# Importing the libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the classifier 
classifier = Sequential()
#step 1 Convolution
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation = 'relu'))
#step 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#step 3 Flattening
classifier.add(Flatten())
#step 4 Full connection
classifier.add(Dense(units = 128, activation = 'relu')) #units means output_dim
classifier.add(Dense(units = 1, activation = 'sigmoid'))
#compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# =============================================================================
#                            PART 2
# =============================================================================
# Importing the required classes or libraries
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64), #based on input_shape = ()
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
#==============================================================================
# Saving the trained model and then importing or loading the model
# =============================================================================
from keras.models import model_from_json
# serialize with JSON
model_json = classifier.to_json()
with open('savedModel.json', 'w') as json_file:
    json_file.write(model_json)
# serialize with HDF5
classifier.save_weights('savedmodel.h5')
print("Saved to disk")
#==============================================================================
# Making New Predictions
# =============================================================================
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
