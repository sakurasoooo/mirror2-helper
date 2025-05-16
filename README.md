# mirror2-helper

## Description
`mirror2-helper` is an automation tool designed to play the game "Mirror2". It captures the game window, analyzes the game board using image processing and a machine learning model, determines optimal moves, and executes them by simulating user input.

## Features
* **Game Automation**: Automatically plays the "Mirror2" game.
* **Window Capture**: Captures the game window for analysis. Utilizes Windows Graphics Capture API (`rt.py`) and potentially other methods via an external `Gamewindow` module.
* **Board Recognition**: Employs OpenCV for image processing tasks like edge detection and grid identification to understand the game board structure.
* **Game Element Classification**: Uses a Keras/TensorFlow based machine learning model (`mirrormodel.py`) to identify different types of game pieces (e.g., 'blue', 'green', 'purple', 'red', 'riya', 'yellow').
* **Move Solving**: Implements algorithms to find valid and optimal moves on the game board.
* **Input Simulation**: Simulates mouse clicks and movements to interact with the game.
* **Custom Model Training**: Includes utilities in `mirrormodel.py` to train your own image classification model for recognizing game elements.

## Core Components
* **`main.py`**: The main script that orchestrates the entire bot logic. It handles capturing the game screen, processing the image, predicting board state, deciding on moves, and executing them.
* **`mirrormodel.py`**: Manages the machine learning aspects. This includes loading a pre-trained Keras model for classifying game pieces and provides functions to train a new model from a dataset of images.
* **`rt.py`**: Implements screen capture functionality using the Windows Graphics Capture API for efficient frame grabbing.
* **`win32key.py`**: Defines constants for keyboard keys, used in simulating inputs.
* **`gamewindow.py` (External/Missing)**: This file is imported by `main.py` for game window management (e.g., finding the window, resetting its size, and capturing). It appears to be a necessary component that is not included in the provided file list.

## Key Technologies & Dependencies
This project relies on several Python libraries. You'll need Python 3 installed.

* **Core Logic & Automation**:
    * Python 3
* **Machine Learning**:
    * TensorFlow (Keras API)
* **Image Processing & Numerics**:
    * OpenCV (`opencv-python`)
    * NumPy
    * Pillow (PIL)
* **Screen Capture & Windows Interaction**:
    * `python-mss` (for screen capture, indicated by `method='mss'`)
    * `pywin32` (for Windows API interaction, including mouse/keyboard simulation using constants from `win32key.py`)
    * `winrt` (specifically modules like `Windows.Graphics.Capture`, used in `rt.py`)
* **Asynchronous Operations**:
    * `asyncio` (used in `rt.py`)
* **Utilities/Plotting (for development/training)**:
    * Matplotlib

## Setup
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd mirror2-helper-main
    ```
2.  **Install Python 3**: Ensure Python 3.7+ is installed.
3.  **Install dependencies**:
    ```bash
    pip install tensorflow opencv-python numpy pillow matplotlib mss pywin32 winrt
    ```
    *Note: Depending on your system, `pywin32` might require installation via `pypiwin32` or specific build tools.*
4.  **`Gamewindow` Module**: The `Gamewindow` class/module imported in `main.py` is not provided. You will need to ensure this module is available in your Python path or implement its functionality. This module is expected to handle tasks like finding the "Mirror2" game window, potentially resizing it, and providing a capture method.
5.  **ML Model**:
    * The bot expects a trained Keras model in a directory named `mirror2_model`.
    * If you have a pre-trained model, place it in the root directory of the project.
    * Alternatively, you can train your own model using the instructions in the "Model Training" section below.

## How to Run
1.  **Start the Game**: Launch the "Mirror2" game and ensure it is visible on your screen.
2.  **Run the Bot**: Execute the main script from the project's root directory:
    ```bash
    python main.py
    ```
3.  The script will attempt to:
    * Find and reset the "Mirror2" game window.
    * Continuously capture the game screen.
    * Process the captured image to identify the game board and pieces.
    * Determine the best move.
    * Simulate mouse actions to perform the move.
    * Repeat the process.

## Model Training (Optional)
If you wish to train your own model for recognizing game pieces:
1.  **Prepare Dataset**:
    * Collect images of individual game pieces from "Mirror2".
    * Organize them into subdirectories named after their class (e.g., `blue`, `green`, `purple`, `red`, `riya`, `yellow`).
    * Place these class directories inside `captures/mirror2/training/`.
2.  **Run Training Script**:
    * The `mirrormodel.py` file contains a `training_model()` function.
    * You might need to modify `mirrormodel.py` to execute this function directly (e.g., by adding `if __name__ == '__main__': training_model()` at the end) or call it from another script.
    * Executing this function will train a new model based on your dataset and save it (likely as `mirror2_model`).

## Disclaimer
* This tool is intended for educational and experimental purposes only.
* Using automation tools or bots might be against the Terms of Service of "Mirror2" or other games. Use this software responsibly and at your own risk. The developers are not responsible for any consequences of using this tool.
