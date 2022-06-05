import torch
import torch.nn as nn

# class MyEnsemble(nn.Module):
#     def __init__(self, modelA, modelB):
#         super(MyEnsemble, self).__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.classifier = nn.Linear(4, 2)

#     def forward(self, x1, x2):
#         x1 = self.modelA(x1)
#         x2 = self.modelB(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.classifier(F.relu(x))
#         return x

class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, input_size, output_size):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, xA, xB):
        out1 = self.modelA(xA)
        out2 = self.modelB(xB)

        # out = out1 + out2
        out = torch.cat((out1, out2), dim=1)

        x = self.classifier(out)
        return torch.softmax(x, dim=1)