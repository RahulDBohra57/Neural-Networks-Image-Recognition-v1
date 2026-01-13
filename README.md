## Fashion-MNIST Image Recognition App
CNN-Powered Clothing Classification Web App

📌 Project Overview

This project is an end-to-end deep learning application that classifies clothing images into one of 10 fashion categories using a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset.
Users can upload images (available in the dataset folder) through a Streamlit web interface, and the model predicts the clothing type along with confidence scores.

The project demonstrates how a computer-vision model can be trained, evaluated, exported, and deployed as a real-world AI web application.

🧠 Dataset

The Fashion-MNIST dataset is a widely used benchmark in computer vision. It contains:
- 70,000 grayscale images
- 28×28 pixels
- 10 clothing categories

Label	Class:
0	T-shirt / Top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle Boot

⚙️ Model Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fbbd41ed-b5a7-448c-9285-5270724c9aff" />

The application uses a CNN (Convolutional Neural Network) optimized for Fashion-MNIST

🖥 Web Application

The trained CNN model is deployed using Streamlit, allowing users to:
Upload a clothing image
View the uploaded image
Receive predicted clothing category
See model confidence scores
This turns the ML model into an interactive AI tool.


🎯 Key Features

- CNN-based fashion classification
- Real-time image upload and prediction
- Class probability visualization


🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pillow
- Scikit-Learn

📌 Disclaimer

This project is built for educational and portfolio purposes.
The model is trained on Fashion-MNIST and performs best on images with similar visual style.
