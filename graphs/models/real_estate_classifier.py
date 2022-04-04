import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.batch_norm = nn.BatchNorm1d(num_classes, momentum=0.99, eps=1e-3)
        
        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.pretrained_model(input)
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)