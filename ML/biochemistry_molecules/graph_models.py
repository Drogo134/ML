import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import Config

class GCNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.3):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return self.classifier(x)

class GATModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 8, num_layers: int = 3, dropout: float = 0.3):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return self.classifier(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, output_dim: int,
                 nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch):
        x = self.input_projection(x)
        
        batch_size = batch.max().item() + 1
        max_nodes = x.size(0)
        
        x_padded = torch.zeros(batch_size, max_nodes, self.d_model, device=x.device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            node_mask = (batch == i)
            node_count = node_mask.sum().item()
            x_padded[i, :node_count] = x[node_mask]
            mask[i, :node_count] = True
        
        x_transformed = self.transformer(x_padded, src_key_padding_mask=~mask)
        
        x_pooled = []
        for i in range(batch_size):
            node_mask = (batch == i)
            node_count = node_mask.sum().item()
            if node_count > 0:
                x_pooled.append(x_transformed[i, :node_count].mean(dim=0))
            else:
                x_pooled.append(torch.zeros(self.d_model, device=x.device))
        
        x_pooled = torch.stack(x_pooled)
        return self.classifier(x_pooled)

class MolecularGraphTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
    def create_data_loader(self, graphs: List[Dict], targets: np.ndarray, 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        data_list = []
        
        for i, (graph_data, target) in enumerate(zip(graphs, targets)):
            if graph_data is None:
                continue
                
            data = Data(
                x=torch.tensor(graph_data['node_features'], dtype=torch.float32),
                edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(graph_data['edge_features'], dtype=torch.float32),
                y=torch.tensor(target, dtype=torch.float32),
                batch=torch.tensor([i] * len(graph_data['node_features']), dtype=torch.long)
            )
            data_list.append(data)
        
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, model_name: str, train_loader: DataLoader, 
                   val_loader: DataLoader, task_type: str = 'classification'):
        if model_name not in self.config.MODEL_PARAMS:
            raise ValueError(f"Model {model_name} not found in config")
        
        params = self.config.MODEL_PARAMS[model_name]
        
        if model_name == 'gcn':
            model = GCNModel(
                input_dim=7,
                hidden_dim=params['hidden_dim'],
                output_dim=1 if task_type == 'classification' else 1,
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_name == 'gat':
            model = GATModel(
                input_dim=7,
                hidden_dim=params['hidden_dim'],
                output_dim=1 if task_type == 'classification' else 1,
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        elif model_name == 'transformer':
            model = TransformerModel(
                input_dim=7,
                d_model=params['d_model'],
                output_dim=1 if task_type == 'classification' else 1,
                nhead=params['nhead'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index, batch.batch)
                
                if task_type == 'classification':
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y)
                else:
                    loss = F.mse_loss(out.squeeze(), batch.y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    
                    if task_type == 'classification':
                        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y)
                    else:
                        loss = F.mse_loss(out.squeeze(), batch.y)
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.config.MODELS_DIR / f"{model_name}_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        model.load_state_dict(torch.load(self.config.MODELS_DIR / f"{model_name}_best.pth"))
        return model
    
    def evaluate_model(self, model_name: str, test_loader: DataLoader, task_type: str = 'classification'):
        model = self.models[model_name]
        model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.batch)
                
                if task_type == 'classification':
                    pred = torch.sigmoid(out.squeeze())
                else:
                    pred = out.squeeze()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if task_type == 'classification':
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = (predictions_binary == targets).mean()
            return {'accuracy': accuracy, 'predictions': predictions, 'targets': targets}
        else:
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
            return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': predictions, 'targets': targets}
