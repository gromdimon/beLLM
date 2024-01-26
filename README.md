# beLLM

![header](assets/header.png)


## Introduction

beLLM is the first 


## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Getting Started](#getting-started)
4. [License](#license)
5. [Contact](#contact)


## Project Overview

In this repository, we focus on reimplementing a range of machine learning projects, ranging from foundational algorithms to contemporary AI solutions. The goal is to deepen our understanding of these algorithms, improve upon them, and share insights with the broader community.

### Projects Included:
- Mikrograd: A minimal autograd engine based on [micrograd](https://github.com/karpathy/micrograd) by `Andrej Karpathy`
- Makemore: A tool for generating synthetic data. Based on the [makemore](https://github.com/karpathy/makemore/tree/master) by `Andrej Karpathy`
- RuPoemGPT: A character-level language model trained on a collection of russian poems. Based on the [nanoGPT](https://github.com/karpathy/nanoGPT) by `Andrej Karpathy`

  **Also here are some older projects, which were not included into this repository:**
- [Glycoprotein prediction with AlphaFold2](https://github.com/gromdimon/AlphaFold_Glycoprotein)
- [Analysis of Long COVID Symptoms using BERT and Twitter Data](https://github.com/gromdimon/LongCovid)
- [Compound bioactivity prediction](https://github.com/gromdimon/Bioactivity_prediction_project)
- [General Machine Learning Techniques practice](https://github.com/gromdimon/Training--Projects)


## Technologies Used

- **Python**: The primary programming language used for implementing algorithms.
- **Libraries/Frameworks**: PyTorch, Scikit-learn, Matplotlib.


## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Ensure you have `pyenv` and `pipenv` installed on your system. If not, follow these steps:

- Install `pyenv`:
  ```
  curl https://pyenv.run | bash
  ```

- Install `pipenv`:
  ```
  pip install pipenv
  ```

### Installation

Follow these steps to get your development environment running:

1. Clone the repository:
   ```
   git clone https://github.com/gromdimon/ml-random.git
   ```

2. Navigate to the project directory:
   ```
   cd ml-random
   ```

3. Set the local Python version using `pyenv`:
   ```
   pyenv install 3.10.12
   pyenv local 3.10.12
   ```

4. Install `pipenv` for the local Python version:
   ```
   pip install --user pipenv
   ```

5. Install dependencies using `pipenv`:
   ```
   pipenv install
   ```

6. Activate the `pipenv` environment:
   ```
   pipenv shell
   ```

### Project-Specific Instructions

Each project within this repository may have additional setup or instructions. Please refer to the README.md file within each project's directory for more specialized guidance.


## License

Distributed under the MIT License. See `LICENSE` for more information.
