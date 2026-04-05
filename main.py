import argparse
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch
from model import GNNModel

def load_data(raw=False) -> DataLoader:
    print("Loading QM9 dataset for molecules...")
    # Load the QM9 dataset
    # The dataset will be automatically downloaded to './data/QM9' on first run
    dataset = QM9(root='./data/QM9')
    
    print(f"Dataset loaded successfully!")
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of features per molecule: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Display information about the first molecule
    if len(dataset) > 0:
        data = dataset[0]
        show_data_info(data)
    
    # Create a DataLoader for batching
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"\nCreated DataLoader with batch size: {batch_size}")
    print("QM9 dataset import completed successfully!")
    return loader

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

def main():
    # if argument --raw is passed, use raw data
    parser = argparse.ArgumentParser(description="GNN for Molecular Property Prediction")
    parser.add_argument('--raw', action='store_true', help="Show raw data information and exit")
    args = parser.parse_args()

    data_loader = load_data(raw=args.raw)

    # import the model
    model = GNNModel(in_channels=11, hidden_channels=64, out_channels=1)
    print("\nGNN Model initialized successfully!")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        # Predict
        out = model(data.x, data.edge_index, data.batch)
        # QM9 has 19 targets. Let's predict the first one (index 0)
        loss = criterion(out, data.y[:, 0].view(-1, 1))
        # Backprop
        loss.backward()
        optimizer.step()
    return loss.item()



if __name__ == "__main__":
    main()
