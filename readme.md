# Deep Learning
Assignment for Deep and Reinforcement Learning Class, 1º Year,2º Semester, Masters in Artificial Intelligence 

# Summary

In this project, our objective was to develop a **Discriminator** capable of classifying images as either real or fake, and a **Generator** that can produce synthetic images resembling those in the original dataset.

For training the Discriminator, we utilized the [DeepFakeFace dataset](https://huggingface.co/datasets/OpenRL/DeepFakeFace), which contains a wide range of real and manipulated facial images. To train the Generator, we employed the [CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download), known for its extensive collection of celebrity face images with rich facial attribute annotations.


**Authors**:
- [Sebastião Santos Lessa](https://github.com/seblessa/)
- [Guilherme Vaz](https://github.com/guilhermevaz8)

# Versões

The versions of the operating systems used to develop and test this application are:
- Ubuntu 24.04.2 LTS x86_64

Python Versions:
- 3.12

# Requirements

To keep everything organized and simple, we will use MiniConda to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n venv python=3.12
```
To install the requirements run:

```bash
pip install -r requirements.txt
```

# Report

You can see the notebook here: [notebook.ipynb](notebook.ipynb).
