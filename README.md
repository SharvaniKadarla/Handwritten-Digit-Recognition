
# Handwritten Digit Recognition

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset using PyTorch. It is modularized into separate Python files simulated using %%writefile in Google Colab, mimicking a real-world deep learning project structure.

# 📁 Project Structure

```text
├── model.py           # Defines the CNN architecture
├── data_loader.py     # Loads MNIST dataset using DataLoader
├── utils.py           # Utility to detect available device (CPU/GPU)
├── train.py           # Training loop, model saving, loss plotting
├── evaluate.py        # Loads model and performs digit prediction
├── model_state.pt     # Saved model weights
├── training_loss_curve.png  # Visualization of training loss
├── Handwritten Digit Images/
│   ├── digit1.png               # Sample digit image
│   ├── digit2.png
│   └── ...                      # Other digit images used for evaluation
```

# 🚀 Installation
Install the required libraries in your Python environment:
```text
pip install torch torchvision Pillow
```

# 📦 Dataset
We use the MNIST handwritten digits dataset, which is automatically downloaded using ```torchvision.datasets.MNIST```.

# 🏗️ Model Architecture
The CNN defined in ```model.py``` has:

- 3 convolutional layers with ReLU activation

- A final fully connected (```Linear```) layer that outputs 10 class scores (digits 0–9)

- Uses ```CrossEntropyLoss``` for training

# 🏋️ Training
To train the model: 

```python train.py```

This will:

- Train for 5 epochs (can be customized)

- Save the model weights to ```model_state.pt```

- Generate and save a loss curve as ```training_loss_curve.png```

# 🔍 Evaluation
Make sure:

- The image is grayscale (```.png``` format)

- The digit is dark on a light background (or use ```ImageOps.invert()```)

📁 Note: All test images like ```digit1.png```, ```digit2.png```, etc. are stored in the ```Handwritten Digit Images/``` folder.

# 📊 Output Example
- A matplotlib window displays the input image with the predicted label.

- The predicted label is also printed in the console.

# 🛠️ Notes
- Files are created inside Google Colab using ```%%writefile```, simulating a structured project.

- Model weights are saved as ```model_state.pt```.

- Training visualizations (loss curves) help assess model performance over epochs.





