""" Imports """
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.datasets import fashion_mnist
from PIL import Image

class NeuralNetwork:
    """ Class Neural Network"""
    def __init__(self, class_names):
        self.class_names = class_names

    def __str__(self):
        """ String representation"""
        return (f" Neural Network;\n"
                f"class names {self.class_names}")
    def __repr__(self):
        """ Technical representation"""
        return f"NeuralNetwork({self.class_names})"

    def load_prepare_data(self):
        """ Load and prepare the data of fashion mnist"""

        (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
        train_x = train_x / 255.0 # normalize the data
        test_x = test_x / 255.0 # normalize the data

        return train_x, test_x, train_y, test_y

    def visualize(self, image, labels):
        """
        Visualize the train data set
        :input; train image, train labels

        """
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image[i], cmap=plt.cm.binary)
            plt.xlabel(f"{self.class_names[labels[i]]}", color='purple')
        plt.show()

    def train_model(self, train_x, train_y):
        """
        Train the model
        :input train_x: images train dataset
               train_y: labels train dataset
        """
        (x_train, x_test,
         y_train, y_test) = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        model = keras.Sequential([
            # Transforms images two dimenstional to one dimensional
            keras.layers.Flatten(input_shape=(28, 28)),
            # First layer 128 neuronen
            keras.layers.Dense(128, activation='relu'),
            # Second layer length of 10 (10 classes)
            keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1) # Fits the model with epochs
        loss, accuracy = model.evaluate(x_test, y_test)  # evaluates the model

        print(f"Test accuracy: {accuracy:.3f}")  # Shows the test accuracy
        print(f"Test loss: {loss:.3f}")  # Shows the test loss
        return model, x_test, y_test

    def predict(self, model, x_test, y_test, num_rows, num_cols):
        """ Predict the image label and visualize it """

        predictions = model.predict(x_test)
        # print(predictions[0])
        predicted_labels = np.argmax(predictions, axis=1)
        plt.figure(figsize=(10, 10))
        num_images = num_rows * num_cols

        for i in range(num_images):
            plt.subplot(num_rows,2*num_cols,2*i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_test[i], cmap='gray')
            if predicted_labels[i] == y_test[i]:
                color = 'blue'
            else:
                color = 'purple'
            plt.xlabel(f"{self.class_names[predicted_labels[i]]} (0{y_test[i]})", color=color)
            plt.subplot(num_rows,2*num_cols,2*i+2)
            plt.grid(False)
            plt.xticks(range(10), self.class_names, rotation=90, fontsize=8)
            thisplot = plt.bar(range(10), predictions[i], color="green")
            plt.ylim([0, 1])
            thisplot[np.argmax(predicted_labels[i])].set_color('blue')
            thisplot[y_test[i]].set_color('purple')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    nn = NeuralNetwork(class_names)
    x_train, x_test, y_train, y_test = nn.load_prepare_data()
    #nn.visualize(x_train, y_train) # uncomment it to see the traindataset visualize
    model, test_img, test_label = nn.train_model(x_train, y_train)
    img_path = 'data/vernon.jpg'
    # input()
    # change it to your own image path
    img = Image.open(img_path)
    img = img.convert(mode='L')
    new_size = (28, 28)
    img = img.resize(new_size)
    img = np.array(img)
    nn.predict(model, img[None,:,:], y_test, 1, 1)
    #nn.predict(model, x_test, y_test, 5, 5) # you can use this to predict with the test data set
