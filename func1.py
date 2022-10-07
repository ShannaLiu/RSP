# Directly minimize the loss function in pytorch framework
import torch

class CURSolver(torch.nn.Module):
    def __init__(self, N, alpha, lamda):
        super().__init__()
        # Initialize W with uniform distribution
        self.W = torch.nn.Parameter(torch.Tensor(N,N))
        torch.nn.init.uniform_(self.W, -1/N, 1/N)
        self.alpha = alpha 
        self.lamda = lamda

    def forward(self, X, T):
        error = X - torch.matmul(self.W, X)
        loss = torch.sum(error.pow(2)) + self.alpha * torch.sum(torch.sum(self.W.pow(2), dim=1).sqrt()) + \
            self.lamda * torch.sum((T*self.W).abs())
        return loss


def CUR_torch_solver(X, T, alpha, lamda, epochs, learning_rate, print=False):
    N = X.shape[0]
    model = CURSolver(N, alpha, lamda)
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




