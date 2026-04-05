# GNN for Molecular Property Prediction

This project implements a Graph Neural Network (GNN) using PyTorch and PyTorch Geometric to predict molecular properties from the QM9 dataset. Specifically, the model is trained to predict the dipole moment (μ) of molecules.

The `main.py` script orchestrates the entire pipeline, including data loading, splitting into training and testing sets, model training, evaluation, and report generation. The GNN architecture is defined in `model.py`.

## Adjustments
QM9 includes data about molecules that are generally small. The first model implementation used only two GCNConv layers, which may not be sufficient for capturing the complex relationships in molecular graphs. 
When we are working with molecules with deeper paths, we need to allow the model to learn about neighbors of neighbors (and even further).
To address this, we can add more additional GCNConv layers to allow the model to learn.
It's good to keep in mind that adding too many layers can lead to overfitting (in this case also called oversmoothing, allowing the system to coverge to a stable vector for molecules), especially with a small dataset like QM9. 
Adding more input features (like mass, charge, etc.) doesn't require more layers, but it might require "wider" layers (e.g., changing hidden_channels from 64 to 256).
It is recommended to experiment with the number of layers and hidden channels to find the best configuration for your specific task.
The initial implementation of the model uses a simple global mean pooling layer to aggregate node features into a graph-level representation.
This is a common approach, but it may not capture all the relevant information from the graph.
To improve this, we can experiment with different pooling methods (e.g., global max pooling, attention-based pooling, etc.) or even use a combination of pooling methods to capture different aspects of the graph structure.
My intuition would be to also follow the deep learning practices like normalization layers, deeper lineayr layers for logical systems and wider layers for more complex systems, and regularization techniques like dropout to prevent overfitting.
This model implementation is a good baseline. To get better predictions you'd likely use Edge features like GINEConv, or even more advanced architectures like Graph Attention Networks (GAT) or Message Passing Neural Networks (MPNNs) instead of GCNConv.
Also, the distance is a very important feature in molecular graphs, and using a model that can incorporate edge features (like distances) would likely improve performance significantly. we could improve the model by using Radial Basis Functions (RBF) as a GNN layer that can incorporate edge features, such as GINEConv or MPNN, instead of GCNConv. This would allow the model to take into account the distances between atoms, which is crucial for molecular property prediction.
For the future we could use Attention Mechanisms to allow the model to focus on the most relevant parts of the graph when making predictions. This could be implemented using Graph Attention Networks (GAT) or by adding attention layers on top of the existing GCN layers. Another improvement would be incorporating equivariant layers to handle the 3D nature of molecular data, which can help the model learn more effectively from the spatial relationships between atoms. Additionally, we could experiment with different loss functions and optimization techniques to further improve the model's performance on the QM9 dataset.

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
