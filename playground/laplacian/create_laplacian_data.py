import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def main():
    # Load dataset
    dataset = LRGBDataset(root='data/LRGBDataset', name='Peptides-struct')
    if dataset:
        print("dataset was successfully loaded")

    # Define the Laplacian PE transform
    num_eigenvectors = 3
    pe_transform = AddLaplacianEigenvectorPE(k=num_eigenvectors, attr_name="lap_pe")

    transformed_dataset = []
    # Apply the transformation to each data object and concatenate Laplacian PE
    for data in dataset:
        data = pe_transform(data)
        data.x = torch.cat([data.x, data.lap_pe], dim=1)
        transformed_dataset.append(data)

    # Save the transformed dataset
    torch.save(transformed_dataset, "/data/peptide_struct-transformed.pt")
    print("Finished Transformation")


if __name__ == "__main__":
    main()