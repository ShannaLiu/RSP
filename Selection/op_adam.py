# Optimization - torch 
# Directly minimize the loss function in pytorch framework
import torch

class CURSolver(torch.nn.Module):
    def __init__(self, N, alpha, lamda, method):
        super().__init__()
        # Initialize W with uniform distribution
        self.W = torch.nn.Parameter(torch.Tensor(N,N))
        torch.nn.init.uniform_(self.W, -1/N, 1/N)
        self.alpha = alpha 
        self.lamda = lamda
        if method in ['Lap', 'dist']:
            self.method = method
        else:
            print('Wrong regularizer, use lap as default')
            self.method = 'Lap'

    def forward(self, X, T):
        # T is the distance matrix or Laplacian matrix
        error = X - torch.matmul(self.W, X)
        if self.method == 'Lap':
            regularizer = torch.sum( torch.matmul(torch.matmul(self.W.t(), T), self.W).pow(2)  )
        else: 
            regularizer = torch.sum((T*self.W).abs())
        loss = torch.sum(error.pow(2)) + self.alpha * torch.sum(torch.sum(self.W.pow(2), dim=1).sqrt()) + \
            self.lamda * regularizer
        return loss


def CUR_adam_solver(X, T, alpha, lamda, method, epochs, learning_rate, print=False):
    N = X.shape[0]
    model = CURSolver(N, alpha, lamda, method)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        loss = model(X, T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        loss = model(X,T)
        if print:
            if i % 10 == 1: 
                print(f'epoch={i} loss={loss:.4f}')
    return model.W, loss




