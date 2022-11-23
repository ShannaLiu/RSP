# A baseline GCN model

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
def train_test_model(data, num_features, hidden_channels, num_classes, epochs, lr, weight_decay, train_mask, test_mask):
    model = GCN(num_features, hidden_channels, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)  
        loss = criterion(out[train_mask], data.y[train_mask]) 
        loss.backward()  
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  
    test_correct = (pred[test_mask] == data.y[test_mask])
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
    print(f'Final accuracy on test set is : {test_acc:.3f}')
    return test_acc



        



