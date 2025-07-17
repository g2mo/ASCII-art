# Generative ASCII Art Creator

This Python script uses a pre-trained deep learning model (VGG19) to analyze the features of an image and generate a stylized ASCII art representation.

## How to Use

1.  **Add Images:**
    Place any `.png` images you want to convert into the same folder as the `generative_ascii.py` script.

2.  **Run the Script:**
    Execute the script from your terminal or IDE:
    `python generative_ascii.py`

The script will automatically find all `.png` files in the directory and create a corresponding `ASCII_your-image-name.txt` file for each one.

## Configuration

You can change the script's behavior by editing these variables directly in the `generative_ascii.py` file:

* `light_mode`:
    * Set to `True` if you will view the output on a **light background** (like a white text editor).
    * Set to `False` if you will view it on a **dark background** (like a black terminal).

* `output_width`:
    * Sets the width of the generated ASCII art in characters. A value around `120` is a good starting point.
