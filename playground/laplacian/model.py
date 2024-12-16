import torch
from torch_geometric.nn import global_mean_pool, GCN


class newGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        # Initialize Superclass
        super().__init__()

        # GCN layers
        self.gcn = GCN(in_channels=in_channels, 
                       hidden_channels=hidden_channels, 
                       out_channels=hidden_channels,
                       num_layers=6, 
                       act='relu', 
                       dropout=0.1, 
                       norm='batch', 
                       norm_kwargs={'track_running_stats': False})

        self.linear = torch.nn.Linear(hidden_channels, out_channels)

        # Multi-layer prediction head
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_channels, hidden_channels),
            # torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),

            torch.nn.Linear(hidden_channels, out_channels)
        )


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # pass trough GCN
        x = self.gcn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        print(f"After GCN: {x.shape}")
        
        # perform global pooling
        x = global_mean_pool(x, batch)
        print(f"After global pooling: {x.shape}")

        # pass through prediciton head
        x = self.prediction_head(x)
        print(f"After prediction head: {x.shape}")

        return x