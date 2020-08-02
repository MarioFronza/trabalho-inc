import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (AveragePooling2D, Dense, Dropout, Flatten,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from tensorflow.keras.utils import to_categorical


class MaskDetectorModel:
    def __init__(self):
        self.learning_rate = 1e-4
        self.number_of_epochs = 15
        self.bs = 32

        self.data = []
        self.labels = []

        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

        self.augmentation = {}
        self.base_model = {}
        self.network_model = {}

    def start(self):
        self.read_images_from_dataset()
        self.define_training_and_testing_sets()
        self.define_training_image_generator()
        self.define_model_head()
        self.compile_model()
        self.train_head()
        self.evaluate_network()

    def read_images_from_dataset(self):
        imagePaths = list(paths.list_images("dataset"))
        label_binarizer = LabelBinarizer()
        for imagePath in imagePaths:
            [pathStart, _] = imagePath.split("-")
            label = pathStart.split("/")[-2]

            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            self.data.append(image)
            self.labels.append(label)

        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)
        self.labels = label_binarizer.fit_transform(self.labels)
        self.labels = to_categorical(self.labels)

    def define_training_and_testing_sets(self):
        (self.train_x, self.test_x, self.train_y, self.test_y) = train_test_split(
            self.data, self.labels, test_size=0.20, stratify=self.labels, random_state=42)

    def define_training_image_generator(self):
        self.augmentation = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        self.base_model = MobileNetV2(weights="imagenet", include_top=False,
                                      input_tensor=Input(shape=(224, 224, 3)))

    def define_model_head(self):
        head_model = self.base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(128, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(2, activation="softmax")(head_model)
        self.network_model = Model(
            inputs=self.base_model.input, outputs=head_model)
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile_model(self):
        network_options = Adam(lr=self.learning_rate,
                               decay=self.learning_rate / self.number_of_epochs)
        self.network_model.compile(loss="binary_crossentropy", optimizer=network_options,
                                   metrics=["accuracy"])

    def train_head(self):
        self.network_model.fit(
            self.augmentation.flow(
                self.train_x, self.train_y, batch_size=self.bs),
            steps_per_epoch=len(self.train_x) // self.bs,
            validation_data=(self.test_x, self.test_y),
            validation_steps=len(self.test_x) // self.bs,
            epochs=self.number_of_epochs)

    def evaluate_network(self):
        self.network_model.predict(self.test_x, batch_size=self.bs)
        self.network_model.save("model", save_format="h5")


if __name__ == "__main__":
    model = MaskDetectorModel()
    model.start()
x
