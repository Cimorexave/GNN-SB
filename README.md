# GNN for Molecular Property Prediction

This project implements a Graph Neural Network (GNN) using PyTorch and PyTorch Geometric to predict molecular properties from the QM9 dataset. Specifically, the model is trained to predict the dipole moment (μ) of molecules.

The `main.py` script orchestrates the entire pipeline, including data loading, splitting into training and testing sets, model training, evaluation, and report generation. The GNN architecture is defined in `model.py`.

## How to Run the Project

To set up and run this project, follow these steps:

1.  **Ensure `uv` is installed**: If you don't have `uv` (a fast Python package installer and resolver), you can install it via `pip install uv`.

2.  **Install dependencies**: Navigate to the project root directory and install the necessary packages using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```
    *Note: If `requirements.txt` is not up-to-date, you might need to install `scikit-learn` explicitly: `uv add scikit-learn`.*

3.  **Run the training pipeline**: To train the GNN model and evaluate it on the test set, execute `main.py` with desired arguments. For example, to run for 50 epochs with a batch size of 32:
    ```bash
    uv run python main.py --epochs 50 --batch_size 32
    ```

4.  **Run evaluation only**: If you want to skip training and only evaluate the model (e.g., on a pre-trained model or to generate reports without re-training), use the `--no_train` flag:
    ```bash
    uv run python main.py --no_train
    ```

Upon completion, evaluation results and configuration details will be saved in the `reports/` folder.