# MNIST Handwritten Digit Classifier

Neural network classifier for handwritten digits using TensorFlow/Keras achieving 97.76% accuracy on the MNIST dataset.

## Performance

- **Test Accuracy**: 97.76%
- **Test Loss**: 0.0844
- **Correct Predictions**: ~9,776 out of 10,000 test images

## Dataset

MNIST - Standard handwritten digit database:
- Training images: 60,000
- Test images: 10,000
- Image size: 28x28 pixels
- Pixel values: 0 (black) to 255 (white)
- Classes: 10 (digits 0-9)

## Model Architecture

Fully-connected neural network:

```
Input Layer:    784 neurons (28x28 flattened)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Output Layer:   10 neurons (Softmax activation)
```

**Total Parameters**: 109,386

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Dense (Input) | (None, 128) | 100,480 |
| Dense (Hidden) | (None, 64) | 8,256 |
| Dense (Output) | (None, 10) | 650 |

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: 20%
- **Hardware**: GPU (Google Colab T4)

### Training Progress

| Epoch | Training Accuracy | Validation Accuracy |
|-------|-------------------|---------------------|
| 1 | 80.51% | 94.86% |
| 5 | 98.12% | 97.06% |
| 10 | 99.46% | 97.32% |

Final validation accuracy: 97.39%

## Classification Report

```
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.98      0.99      0.99      1135
           2       0.98      0.97      0.98      1032
           3       0.98      0.97      0.97      1010
           4       0.98      0.98      0.98       982
           5       0.98      0.97      0.97       892
           6       0.98      0.98      0.98       958
           7       0.97      0.98      0.97      1028
           8       0.97      0.97      0.97       974
           9       0.97      0.97      0.97      1009

    accuracy                           0.98     10000
```

## Technologies

- Python 
- TensorFlow 2.19.0
- Keras
- NumPy
- matplotlib
- seaborn
- scikit-learn

## Installation

```bash
git clone https://github.com/varadshajith/mnist-digit-classifier.git
cd mnist-digit-classifier
pip install -r requirements.txt
```

## Usage

```python
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Flatten images
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat = x_test.reshape(-1, 784) / 255.0

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flat, y_train, epochs=10, validation_split=0.2)

# Predict
predictions = model.predict(x_test_flat)
```

## Project Structure

```
mnist-digit-classifier/
├── MNIST.ipynb
├── README.md
└── requirements.txt
```

## Key Insights

The model achieves high accuracy with a simple architecture by:
- Proper normalization (pixel values scaled to 0-1)
- Sufficient hidden layer capacity (128 and 64 neurons)
- Adam optimizer for efficient training
- Softmax activation for multi-class classification

The slight gap between training (99.46%) and validation accuracy (97.32%) indicates minimal overfitting, demonstrating good generalization to unseen data.

## License

MIT License

## Contact

Varad Shajith
- GitHub: [@varadshajith](https://github.com/varadshajith)
- Email: varadshajith@gmail.com
