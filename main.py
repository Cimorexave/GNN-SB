import argparse
import os
import json
from datetime import datetime
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from model import GNNModel

def show_data_info(data):
    # 1. SEE NODE FEATURES (Atom properties)
        print(f"\n=== Node Features (Atom Properties) ===")
        if data.x is not None:
            print(f"  Node features shape: {data.x.shape}")
            print(f"  First 5 atoms' features:")
            for i in range(min(5, data.x.shape[0])):
                print(f"    Atom {i}: {data.x[i].tolist()}")
            print(f"  All node features (first molecule):")
            print(f"    {data.x}")
        else:
            print("  No node features available")
        
        # 2. SEE EDGES AND THEIR PROPERTIES
        print(f"\n=== Edge Information ===")
        if data.edge_index is not None:
            print(f"  Edge index shape: {data.edge_index.shape}")
            print(f"  Edge index (first 10 edges):")
            edges = data.edge_index.t()  # Transpose to get (num_edges, 2)
            for i in range(min(10, edges.shape[0])):
                print(f"    Edge {i}: Atom {edges[i, 0].item()} -> Atom {edges[i, 1].item()}")
            
            # Check for edge attributes
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                print(f"  Edge attributes shape: {data.edge_attr.shape}")
                print(f"  Edge attributes (first 10):")
                for i in range(min(10, data.edge_attr.shape[0])):
                    print(f"    Edge {i}: {data.edge_attr[i].tolist()}")
            else:
                print("  No edge attributes available")
        else:
            print("  No edge information available")
        
        # 3. SEE TARGET PROPERTIES (QM9 quantum chemical properties)
        print(f"\n=== Target Properties (QM9 Quantum Chemical Properties) ===")
        if data.y is not None:
            print(f"  Target shape: {data.y.shape}")
            print(f"  Target properties (19 quantum chemical properties):")
            # QM9 has 19 target properties
            property_names = [
                "mu (Dipole moment)", "alpha (Isotropic polarizability)", 
                "HOMO (Highest occupied molecular orbital)", "LUMO (Lowest unoccupied molecular orbital)",
                "gap (HOMO-LUMO gap)", "R2 (Electronic spatial extent)", 
                "ZPVE (Zero point vibrational energy)", "U0 (Internal energy at 0K)",
                "U (Internal energy at 298.15K)", "H (Enthalpy at 298.15K)",
                "G (Free energy at 298.15K)", "Cv (Heat capacity at 298.15K)",
                "U0_atom (Atomization energy at 0K)", "U_atom (Atomization energy at 298.15K)",
                "H_atom (Atomization enthalpy at 298.15K)", "G_atom (Atomization free energy at 298.15K)",
                "A (Rotational constant A)", "B (Rotational constant B)", 
                "C (Rotational constant C)"
            ]
            for i in range(data.y.shape[1]):
                print(f"    {property_names[i]}: {data.y[0, i].item():.6f}")
        else:
            print("  No target properties available")
        
        # 4. SEE POSITIONAL INFORMATION (3D coordinates)
        print(f"\n=== Position Information (3D Coordinates) ===")
        if hasattr(data, 'pos') and data.pos is not None:
            print(f"  Position shape: {data.pos.shape}")
            print(f"  First 5 atoms' positions (x, y, z):")
            for i in range(min(5, data.pos.shape[0])):
                pos = data.pos[i].tolist()
                print(f"    Atom {i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        else:
            print("  No position information available")
        
        # 5. SEE ALL AVAILABLE ATTRIBUTES
        print(f"\n=== All Available Data Attributes ===")
        for attr_name in dir(data):
            if not attr_name.startswith('_') and not callable(getattr(data, attr_name)):
                attr_value = getattr(data, attr_name)
                if attr_value is not None:
                    if isinstance(attr_value, torch.Tensor):
                        print(f"  {attr_name}: Tensor with shape {attr_value.shape}")
                    else:
                        print(f"  {attr_name}: {attr_value}")

def create_train_test_split(dataset, test_size=0.2, random_state=42):
    """Split dataset into train and test sets"""
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_idx, test_idx

def load_data(test_size=0.2):
    """Load QM9 dataset and create train/test split"""
    print("Loading QM9 dataset for molecules...")
    dataset = QM9(root='./data/QM9')
    
    print(f"Dataset loaded successfully!")
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of features per molecule: {dataset.num_features}")
    
    # Create train/test split
    train_idx, test_idx = create_train_test_split(dataset, test_size=test_size)
    
    # Create subsets
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_idx)} ({len(train_idx)/len(dataset)*100:.1f}%)")
    print(f"  Test samples: {len(test_idx)} ({len(test_idx)/len(dataset)*100:.1f}%)")
    
    return train_dataset, test_dataset

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """Train the model"""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Use first target property (dipole moment)
            target = batch.y[:, 0].view(-1, 1)
            loss = criterion(out, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return train_losses

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y[:, 0].view(-1, 1)
            loss = criterion(out, target)
            
            test_loss += loss.item()
            predictions.extend(out.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return avg_test_loss, mae, rmse, predictions[:10], targets[:10]  # Return first 10 for display

def generate_report(train_losses, test_loss, mae, rmse, model_info, config):
    """Generate report files in reports/ folder"""
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save training metrics
    metrics = {
        "timestamp": timestamp,
        "training_losses": train_losses,
        "final_training_loss": train_losses[-1] if train_losses else None,
        "test_loss": float(test_loss),
        "mae": float(mae),
        "rmse": float(rmse),
        "model_info": model_info,
        "config": config
    }
    
    metrics_file = f"reports/metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 2. Save summary report
    summary_file = f"reports/summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GNN Model Training Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Input channels: {config['in_channels']}\n")
        f.write(f"  Hidden channels: {config['hidden_channels']}\n")
        f.write(f"  Output channels: {config['out_channels']}\n")
        f.write(f"  Learning rate: {config['learning_rate']}\n")
        f.write(f"  Batch size: {config['batch_size']}\n")
        f.write(f"  Epochs: {config['epochs']}\n")
        f.write(f"  Test split: {config['test_size']}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        if train_losses:
            f.write(f"  Final training loss: {train_losses[-1]:.6f}\n")
        else:
            f.write(f"  Final training loss: N/A (no training performed)\n")
        f.write(f"  Test loss: {test_loss:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.6f}\n")
        f.write(f"  Root Mean Square Error (RMSE): {rmse:.6f}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(model_info + "\n")
    
    print(f"\nReports generated:")
    print(f"  - {metrics_file}")
    print(f"  - {summary_file}")
    
    return metrics_file, summary_file

def main():
    parser = argparse.ArgumentParser(description="GNN for Molecular Property Prediction")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--test_size', type=float, default=0.2, help="Test set size ratio")
    parser.add_argument('--no_train', action='store_true', help="Skip training, only evaluate")
    args = parser.parse_args()
    
    # Configuration
    config = {
        'in_channels': 11,  # QM9 has 11 node features
        'hidden_channels': 64,
        'out_channels': 1,  # Predicting one property (dipole moment)
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'test_size': args.test_size
    }
    
    print("=" * 60)
    print("GNN Model Training Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with train/test split
    train_dataset, test_dataset = load_data(test_size=args.test_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = GNNModel(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels']
    ).to(device)
    
    model_info = str(model)
    print(f"\nModel initialized:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    if not args.no_train:
        print(f"\nTraining model for {args.epochs} epochs...")
        train_losses = train_model(model, train_loader, criterion, optimizer, device, args.epochs)
        print(f"Training completed!")
    else:
        print("\nSkipping training (--no_train flag set)")
        train_losses = []
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, mae, rmse, sample_preds, sample_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
    
    print(f"\nSample predictions vs targets (first 10):")
    for i, (pred, target) in enumerate(zip(sample_preds, sample_targets)):
        print(f"  Sample {i}: Pred={pred:.4f}, Target={target:.4f}, Diff={abs(pred-target):.4f}")
    
    # Generate reports
    print("\nGenerating reports...")
    metrics_file, summary_file = generate_report(
        train_losses, test_loss, mae, rmse, model_info, config
    )
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
