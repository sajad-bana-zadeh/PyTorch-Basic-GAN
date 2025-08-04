# PyTorch-Basic-GAN
Basic implementation of a Generative Adversarial Network (GAN) in PyTorch. Trains a generator and discriminator on the MNIST dataset to generate realistic handwritten digit images. Includes model definition, training loop, and image generation visualization. Beginner-friendly tutorial.

# PyTorch-Basic-GAN

Basic implementation of a Generative Adversarial Network (GAN) in PyTorch. This project trains a simple GAN on the MNIST dataset to generate realistic handwritten digit images.

---

## Overview

A GAN consists of two neural networks playing an adversarial game:

- **Generator (G):** Creates fake images from random noise.
- **Discriminator (D):** Tries to distinguish real images (from the dataset) from fake images (produced by the generator).

During training, the generator improves at creating realistic images to fool the discriminator, while the discriminator improves at detecting fakes. The process continues until the generator produces convincing images.

---

## Project Structure

- `generator.py` - Defines the Generator model.
- `discriminator.py` - Defines the Discriminator model.
- `train.py` - Contains the training loop, data loading, and loss calculations.
- `utils.py` (optional) - Utility functions for visualization, saving samples, etc.
- `requirements.txt` - Required Python dependencies.

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your_username/PyTorch-Basic-GAN.git
   cd PyTorch-Basic-GAN
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```
   pip install torch torchvision matplotlib
   ```

---

## Usage

Run the training script:
```
python train.py
```

This will:

- Download and normalize the MNIST dataset.
- Initialize the generator and discriminator models.
- Train the GAN for 25 epochs.
- Print loss values periodically.
- Save generated images every few epochs for visualization.

---

## Key Implementation Details

### 1. Dataset Preparation

- MNIST handwritten digits dataset is normalized to \([-1, 1]\) range.
- DataLoader splits dataset into batches for efficient training.

### 2. Generator Network

- Takes a random noise vector of size 100 as input.
- Uses fully connected layers with ReLU activations.
- Outputs a 28x28 image with pixel values between -1 and 1 (Tanh activation).

### 3. Discriminator Network

- Takes a 28x28 image as input.
- Uses fully connected layers with LeakyReLU activations.
- Outputs a probability (0 to 1) indicating real or fake.

### 4. Loss and Optimization

- Binary Cross Entropy loss is used for both generator and discriminator.
- Adam optimizer with learning rate 0.0002 and betas (0.5, 0.999).

### 5. Training Loop

- Discriminator is trained on both real images (label 1) and fake images (label 0).
- Generator aims to fool discriminator, so its loss is calculated with target label 1 for fake images.
- Losses are backpropagated to update network weights.

---

## Results

Sample generated images will be saved during training to visualize GAN progress. As training proceeds, generated digits should become more realistic.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Inspired by [PyTorch official tutorials](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- Thanks to the PyTorch and deep learning communities for extensive resources.

---

Feel free to fork, experiment, and expand this basic GAN implementation!
```

This README guides users through understanding the GAN, setting up the environment, running the code, and what to expect during training. You can adjust paths or filenames as needed.

[1] https://itwinai.readthedocs.io/latest/tutorials/distrib-ml/torch-tutorial-GAN.html
[2] https://huggingface.co/facebook/ic_gan/blob/main/stylegan2_ada_pytorch/README.md
[3] https://github.com/eriklindernoren/PyTorch-GAN
[4] https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
[5] https://codeocean.com/capsule/1480388/tree/v1
[6] https://git01lab.cs.univie.ac.at/est-gan/pix2pixhd/-/blob/master/README.md
[7] https://replicate.com/rosinality/style-based-gan-pytorch
[8] https://adioshun.gitbooks.io/deep_learning/GAN/
[9] https://sourceforge.net/projects/simple-stylegan2-pyt.mirror/
