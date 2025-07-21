# 🖼️ Image Captioning Using Deep Learning

This project presents a deep learning-based **image captioning system** that can automatically generate natural language descriptions for images. It combines **computer vision** and **natural language processing (NLP)** to understand visual content and express it in coherent textual form.

Image captioning has a wide range of real-world applications including:
- Assisting the visually impaired
- Organizing and tagging images
- Enhancing search engine indexing
- Improving human-computer interaction

---

## 🚀 Project Overview

The project uses a two-part neural network architecture:
1. **Encoder:** A pre-trained Convolutional Neural Network (CNN), like VGG16, extracts features from the input image.
2. **Decoder:** A Recurrent Neural Network (RNN), specifically an LSTM, learns to generate captions based on the extracted image features and the previous words in the sequence.

Together, these networks form an end-to-end image captioning pipeline that can learn from example image-caption pairs and generate descriptions for unseen images.

---

## 🧠 Model Architecture

### 1. **Image Encoder**
- A **pre-trained CNN (VGG16)** is used to convert input images into fixed-size feature vectors (typically 4096-dimensional).
- The final dense layers of the CNN are removed to retain only the feature extraction portion.
- These vectors are reshaped and passed as input to the decoder network.

### 2. **Text Decoder**
- The decoder begins with an **embedding layer** that transforms word indices into dense vectors.
- An **LSTM layer** follows, which processes sequences of embedded words along with image features.
- A **dense output layer** with softmax activation predicts the next word in the sequence, based on context.

### 3. **Merging Strategies**
- The model combines the image vector and embedded textual sequence using either **concatenation** or **dot product**.
- This fused representation is then passed through the LSTM for sequential word prediction.

---

## 🏋️ Training Strategy

The model is trained using **supervised learning**, where the input is:
- **Image feature vector**
- **Partial caption sequence** (e.g., "a man riding")

The target is:
- **Next word in the caption** (e.g., "a man riding" → "bike")

This approach teaches the model to generate one word at a time, conditioned on both the image and the words generated so far.

### Training Details:
- **Loss Function:** Categorical Crossentropy (since it’s a multi-class classification problem over the vocabulary)
- **Optimizer:** Adam
- **Batch Size:** 64 (chosen to balance training speed and memory efficiency)
- **Epochs:** 10
- **Early Stopping:** Used to prevent overfitting by stopping when the validation loss stops improving.

---

## 🧹 Dataset Preparation

### 1. **Captions Dataset**
- Raw captions are collected from an image-caption dataset such as **Flickr8k**.
- Each image has 5 human-written captions that describe the image in different ways.
- Captions are **cleaned** by:
  - Converting to lowercase
  - Removing punctuation and special characters
  - Removing single-character words and numbers
  - Adding `<start>` and `<end>` tokens to help the model learn when to begin and stop generating captions.

### 2. **Image Features**
- Images are preprocessed and passed through a **pre-trained VGG16** model (excluding the classification head).
- The output feature vector is stored and mapped to the corresponding image ID.
- This speeds up training by avoiding repetitive CNN computations.

### 3. **Tokenizer and Vocabulary**
- A Keras tokenizer is fitted on all training captions.
- Each word is assigned a unique index, and captions are converted to integer sequences.
- Sequences are padded to a maximum length for consistency.

---

## 🔁 Data Generator

Due to memory constraints (especially when handling large datasets), a **custom data generator** is used to yield data in batches during training.

It:
- Selects a batch of image-caption pairs
- For each caption, creates multiple input-output samples (partial caption → next word)
- Yields a tuple of `[image_features, padded_input_sequence], output_word`

This generator ensures efficient memory usage and supports training with very large datasets.

---

## 💡 Why Batch Size = 64?

A batch size of 64 is a good trade-off between speed and memory:
- **Larger batch sizes** speed up training due to parallelism but require more GPU memory.
- **Smaller batch sizes** reduce memory usage but increase training time and may lead to noisier gradient updates.
- 64 is commonly used in image-text models as it often fits comfortably within memory limits while maintaining stable training.

---

## 📌 Features

- End-to-end image captioning pipeline using deep learning
- Cleaned and preprocessed dataset for robust model training
- Custom data generator for large-scale batch training
- BLEU score integration for evaluation (optional)
- Modular code design for easy experimentation with different encoders/decoders
