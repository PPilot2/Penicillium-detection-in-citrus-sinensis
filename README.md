# Penicillium Detection in *Citrus sinensis*

This project utilizes a Convolutional Neural Network (CNN) and deep learning techniques to detect *Penicillium* infections in oranges (*Citrus sinensis*). The model processes images of citrus fruits to classify whether they are infected or healthy, providing a fast and efficient way to aid in disease management and reduce agricultural losses.

---

## Features
- **Automated Image Classification**: Detects the presence of *Penicillium* from fruit images.
- **Interactive GUI**: Upload images and get real-time predictions.
- **Performance Metrics**: Displays model accuracy, mean, variance, and standard deviation of training results.
- **Model Saving and Loading**: Trained model is saved locally for reuse.

---

## Technologies Used
- **Framework**: TensorFlow/Keras
- **GUI**: Tkinter
- **Languages**: Python
- **Visualization**: Matplotlib
- **Data Handling**: NumPy, Pandas
- **File Management**: OpenCV, PrettyTable
- **Deep Learning Model**: CNN

---

## Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- Install required libraries:
  ```bash
  pip install tensorflow matplotlib numpy pandas opencv-python-headless prettytable

## To clone the repository
 - git clone https://github.com/yourusername/penicillium-detection.git
 - cd penicillium-detection

 - ## Usage

### Training the Model
- Place your training data in a directory named `data` with subfolders for each class:
  - `Positive` for infected samples.
  - `Negative` for healthy samples.
- Run the script to train the model:
  ```bash
  python your_script.py

