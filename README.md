# Mauritanian Plate Detection Challenge

![](https://user-images.githubusercontent.com/57320216/166916670-03dfabe1-8c6c-471a-875c-8715354aa957.jpg)

This repository presents a solution to the Mauritanian Plate Detection Challenge, where the goal was to train a model to perform OCR on car images and achieve accurate results with a limited number of images. Our model achieved a character-level accuracy of **99.958%**.

There are three directories in this repository:

* **Training**: Contains a Jupyter notebook that explains step by step how I built my solution.

* **Images**: Contains the preprocessed images and a script to automatically preprocess any images given a directory.

* **Predictions**: A folder containing the predictions submitted to the competition.
## Usage/Examples

To make predictions yourself, simply run the `inference.py` script.

It takes two arguments: the directory containing the images you want to process and optionally the path where you want to save the images.

```bash
python inference.py img_dir [--save_path]
```


## Installation

To install the project, use Poetry. Navigate to the directory containing the pyproject.toml file, open the command line interface, and enter:

```cmd
  poetry install
```
    