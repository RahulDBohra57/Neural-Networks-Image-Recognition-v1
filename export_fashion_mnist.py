import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

# ----------------------------------
# Class labels
# ----------------------------------
class_names = [
    "T-shirt_Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_Boot"
]

# ----------------------------------
# Load dataset
# ----------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# ----------------------------------
# Create folder structure
# ----------------------------------
base = "dataset"
train_path = os.path.join(base, "train")
test_path = os.path.join(base, "test")

for cls in class_names:
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

# ----------------------------------
# Export training images
# ----------------------------------
print("Exporting training images...")

for i in range(len(x_train)):
    label = y_train[i]
    class_name = class_names[label]

    img = x_train[i]
    img = Image.fromarray(img)

    img.save(f"{train_path}/{class_name}/{i}.png")

# ----------------------------------
# Export test images
# ----------------------------------
print("Exporting test images...")

for i in range(len(x_test)):
    label = y_test[i]
    class_name = class_names[label]

    img = x_test[i]
    img = Image.fromarray(img)

    img.save(f"{test_path}/{class_name}/{i}.png")

print("Dataset successfully created in ./dataset/")
