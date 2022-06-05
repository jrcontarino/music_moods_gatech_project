
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, num_features, hidden_dim, n_layers, num_class, device = None):
        super(RNN, self).__init__()
        
        if bool(device):
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(num_features, hidden_dim, n_layers, batch_first = True)
        self.layer_out = nn.Linear(hidden_dim, num_class)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.device == 'cuda:0':
            hidden = hidden.cuda()

        out, hidden = self.rnn(x,hidden)
        
        out = self.layer_out(out[:, -1, :]) 
        
        return out