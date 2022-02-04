import torch 
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__() 

        self.cnn1 = CNNBlock(1, 32, )
class CnnModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LstmModel(nn.Module):

    def __init__(self, n_feature, num_classes, n_hidden, n_layers=2, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(n_hidden), int(n_hidden/2))
        self.fc2 = nn.Linear(int(n_hidden/2), num_classes)

    def forward(self, x):

        x = torch.squeeze(x, 1)
        x = x.permute(0,2,1)
        
        state = self.init_state(x.shape[0], device)
        l_out, state = self.lstm(x, state)
        out = self.dropout(l_out)

        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)

        return out
    
    def init_state(self, batch_size, device):
        return (torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device),
                torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device))



class CnnLstm(nn.Module):
    def __init__(self, cnn=None, lstm=None, num_classes=None):
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm 
        self.fc = nn.Linear(20, num_classes)

    
    def forward(self, x):
        out1 = self.cnn(x)
        out2 = self.lstm(x)
        
        out = torch.cat((out1, out2), 1)
        out = self.fc(out)
        # out = x.
        return out


if __name__ == "__main__":
    model = CNNModel()

    x = torch.rand((2, 1,128,172))
    out = model(x)
    print(out.shape)