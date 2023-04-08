import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import pathlib
class_names = ['blue', 'green','purple', 'red','riya', 'yellow']

# class LiteModel:

#     @classmethod
#     def from_file(cls, model_path):
#         return LiteModel(tf.lite.Interpreter(model_path=model_path,num_threads = 24))

#     @classmethod
#     def from_keras_model(cls, kmodel):
#         converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
#         tflite_model = converter.convert()
#         return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

#     def __init__(self, interpreter):
#         self.interpreter = interpreter
#         self.interpreter.allocate_tensors()
#         input_details = self.interpreter.get_input_details()[0]
#         output_details = self.interpreter.get_output_details()[0]
#         self.input_index = input_details["index"]
#         self.output_index = output_details["index"]
#         self.input_shape = input_details["shape"]
#         self.output_shape = output_details["shape"]
#         self.input_dtype = input_details["dtype"]
#         self.output_dtype = output_details["dtype"]

#     def predict(self, img):
#         import time
#         start_time = time.time()
#         height = self.input_shape[1]
#         width = self.input_shape[2]
#         img = img.resize((width, height))
#         img = img.convert('RGB')
#         input_data = np.array(img, dtype=self.input_dtype)
#         input_data = np.expand_dims(input_data, axis=0)
#         self.interpreter.set_tensor(self.input_index, input_data)

#         self.interpreter.invoke()

#         # The function `get_tensor()` returns a copy of the tensor data.
#         # Use `tensor()` in order to get a pointer to the tensor.
#         output_data = self.interpreter.get_tensor(self.output_index)
#         # print(output_data)

#         # score = tf.nn.softmax(output_data[0])
#         score = output_data[0]
#         # print("--- %s seconds ---" % (time.time() - start_time))
#         return np.argmax(score), 100 * np.max(score/255)

class PredictModel:

    @classmethod
    def from_keras_model(cls, kmodel):
        return PredictModel(keras.models.load_model(kmodel))

    def __init__(self, model):
        self.model = model

    def predict(self, img):
        import time
    
        start_time = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height,img_width), interpolation = cv2.INTER_AREA)
        # img = Image.fromarray(img)
        # img = img.convert('RGB')
        # img = img.resize((img_height, img_width), Image.NEAREST)
        # img = tf.keras.utils.load_img(
        #     note_path, target_size=(img_height, img_width)
        # )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model(img_array,training=False)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        print("--- %s seconds ---" % (time.time() - start_time))
        return class_names[np.argmax(score)], 100 * np.max(score)    

work_path = os.path.dirname(__file__)
img_path = os.path.join(work_path, 'captures')
train_path = os.path.join(img_path, 'mirror2', 'training')
data_dir = pathlib.Path(train_path)
# print(data_dir)

batch_size = 20
img_height = 61
img_width = 61

# image_count = len(list(data_dir.glob('*/*.png')))
# print(image_count)

# bear_blue = list(data_dir.glob('bear_blue/*'))
# im = PIL.Image.open(str(bear_blue[1]))
# im.show()
# tulips = list(data_dir.glob('tulips/*'))


    
    
def training_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)


    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))


    num_classes = len(class_names)
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])

    model.summary()



    epochs = 40
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save("mirror2_model")

# star_blue_path = os.path.join(train_path, 'star_blue')
# note_path = os.path.join(star_blue_path, '2.png')

def prediction():
    import time
    
    import io
    model = keras.models.load_model("note_model")
    start_time = time.time()
    
    # print(note_path)


    img = Image.open(note_path)
    img = img.convert('RGB')
    img = img.resize((img_height, img_width), Image.NEAREST)
    # img = tf.keras.utils.load_img(
    #     note_path, target_size=(img_height, img_width)
    # )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model(img_array,training=False)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    return class_names[np.argmax(score)], 100 * np.max(score)
 
# def prediction_lite():
#     import time
    
#     import io
#     kmodel = keras.models.load_model("note_model")
#     lmodel = LiteModel.from_keras_model(kmodel)
#     start_time = time.time()
    
#     # print(note_path)

#     img = Image.open(note_path)
#     img = img.convert('RGB')
#     img = img.resize((img_height, img_width), Image.NEAREST)
    
#     predictions = lmodel.predict(img)
#     score = tf.nn.softmax(predictions[0])

#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )
#     print("--- %s seconds ---" % (time.time() - start_time))
#     # return class_names[np.argmax(score)], 100 * np.max(score)   


# LiteModelPath = './mdmodel/model.tflite'

# def convert_model():
#     # Convert the model
#     converter = tf.lite.TFLiteConverter.from_saved_model("note_model") # path to the SavedModel directory
#     tflite_model = converter.convert()
#     # Save the model.
#     with open(LiteModelPath, 'wb') as f:
#         f.write(tflite_model)


# def lite():
#     # Load the TFLite model and allocate tensors.
#     interpreter = tf.lite.Interpreter(model_path=LiteModelPath)
#     interpreter.allocate_tensors()

#     # Get input and output tensors.
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Test the model on random input data.
#     input_shape = input_details[0]['shape']
#     # check the type of the input tensor
#     # floating_model = input_details[0]['dtype'] == np.float32

#     # NxHxWxC, H:1, W:2
#     height = input_details[0]['shape'][1]
#     width = input_details[0]['shape'][2]
#     img = Image.open(note_path).resize((width, height))
#     img = img.convert('RGB')
#     input_data = np.array(img, dtype=input_details[0]['dtype'])
#     input_data = np.expand_dims(input_data, axis=0)
#     print(input_data.shape)
#     print(interpreter.get_input_details())
#     # img.show()

#     # # add N dim
#     # input_data = np.expand_dims(img, axis=0)

#     # input_mean = input_std = 127.5

#     # # if floating_model:
#     # #     input_data = (np.float32(input_data) - input_mean) / input_std
#     # #     print("NOOOO")
#     # input_data = np.float32(input_data)
#     # input_data = np.array(input_data, dtype=np.float32)
#     interpreter.set_tensor(input_details[0]['index'], input_data)

#     interpreter.invoke()

#     # The function `get_tensor()` returns a copy of the tensor data.
#     # Use `tensor()` in order to get a pointer to the tensor.
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     print(output_data)

#     # score = tf.nn.softmax(output_data[0])
#     score = output_data[0]

#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score/255))
#     )

# def lite_train():

#     from tflite_model_maker import model_spec
#     from tflite_model_maker import image_classifier
#     from tflite_model_maker.config import ExportFormat
#     from tflite_model_maker.config import QuantizationConfig
#     from tflite_model_maker.image_classifier import DataLoader    
#     # Load input data specific to an on-device ML app.
#     data = DataLoader.from_folder(train_path)
#     train_data, rest_data = data.split(0.8)
#     validation_data, test_data = rest_data.split(0.5)

#     # Customize the TensorFlow model.
#     model = image_classifier.create(train_data, validation_data=validation_data, epochs=100,batch_size= 10,dropout_rate= 0.2,shuffle=True,use_augmentation= True)
#     model.summary()
#     loss, accuracy = model.evaluate(test_data)
#     os.makedirs(os.path.join(work_path,'mirrormodel'), exist_ok=True)
#     # Export to Tensorflow Lite model and label file in `export_dir`.
#     model.export(export_dir=os.path.join(work_path,'mirrormodel'), export_format=[ExportFormat.TFLITE,ExportFormat.LABEL,ExportFormat.SAVED_MODEL])
    

# mylitmodel = LiteModel.from_file(LiteModelPath)
# print("PROCCESSING")
# mylitmodel.predict(Image.open(note_path))
# mylitmodel.predict(Image.open(note_path))
# mylitmodel.predict(Image.open(note_path))
# mylitmodel.predict(Image.open(note_path))
# training_model()
# mymodel = PredictModel.from_keras_model('mirror2_model')
# mymodel.predict(cv2.imread(os.path.join(train_path,'purple','1 (6)'+'.jpg')))