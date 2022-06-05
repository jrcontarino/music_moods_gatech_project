import torch
import torch.nn as nn

class MyEnsemble2(nn.Module):

    def __init__(self, modelA, modelB, input_size, output_size):
        super(MyEnsemble2, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        # Remove last linear layer
        self.modelA.layer_out = nn.Identity()
        self.modelB.layer_out = nn.Identity()

        self.classifier = nn.Linear(input_size, output_size)


    def forward(self, xA, xB):
        out1 = self.modelA(xA)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.modelB(xB)
        out2 = out2.view(out2.size(0), -1)

        # out = out1 + out2
        out = torch.cat((out1, out2), dim=1)

        x = self.classifier(out)
        return torch.softmax(x, dim=1)