# iMMage Visual enhanceD
![alt text](image.png)

This is a web application built with **Flask** that uses a **Super-Resolution Convolutional Neural Network (SRCNN)** to upscale low-resolution images. Users can upload an image, and compare the original lower-quality version with the enhanced high-resolution result.

## Features

*   **Image Upscaling**: Uses a deep learning model (SRCNN) to increase image resolution by 4x.
*   **Comparison View**: Toggle between the "Original" and "Enhanced" versions to see the difference clearly.
*   **Result Download**: Download the processed high-resolution image.


## Installation

1.  **Clone** the repository:
    ```bash
    git clone https://github.com/sleeplessCoderrr/iMMage-Visual-enhanceD.git
    ```
2.  **Navigate** to the project directory:
    ```bash
    cd iMMage-Visual-enhanceD 
    ```
3.  **Install Dependencies**:
    Run command to install required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are on Windows and encounter issues installing PyTorch, please visit [pytorch.org](https://pytorch.org/get-started/locally/) for the specific command for your system.*

## How to Run

1.  Start the Flask application:
    ```bash
    python run.py
    ```
2.  Open your web browser and go to:
    ```
    http://127.0.0.1:5000/
    ```

## Usage

1.  **Upload**: On the home page, drag and drop an image (JPG, PNG) or click to select one.
2.  **Wait**: The app will process the image. Larger images may take a few seconds as the model runs on your CPU.
3.  **Compare**: On the result page, use the toggle switch to flip between the "Original" and "Enhanced" image.
4.  **Download**: Click the "Download Result" button to save the upscaled image.

## Model Information

The application loads a pre-trained SRCNN model from `SRCNN/models/base_srcnn_model.pkl`. The pipeline automatically detects whether to load the standard `SRCNN` or `ImprovedSRCNN` architecture based on the saved state dictionary.