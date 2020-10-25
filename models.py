import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.video import r2plus1d_18
from utils import freeze_all, freeze_layer, freeze_bn, initialize_linear
# torch.backends.cudnn.enabled = False

class R2Plus1D(nn.Module):
    def __init__(self, num_classes=101):
        super(R2Plus1D, self).__init__()

        # r2plus1d_18
        model = r2plus1d_18(pretrained=True)
        
        # encoder(freezing)
        self.encoder_freeze = nn.Sequential(
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.encoder_freeze.apply(freeze_all)

        # encoder(fine-tuning target)
        self.encoder_tune = nn.Sequential(
            model.layer4,
            model.avgpool,
        )
        
        # linear classifier
        self.classifier = nn.Linear(model.fc.in_features, num_classes)
        self.classifier.apply(initialize_linear)

    def forward(self, x):
        b, d, c, h, w = x.shape
        x = x.transpose(1, 2).contiguous() # b, c, d, h, w
        x = self.encoder_freeze(x)
        x = self.encoder_tune(x).squeeze()
        x = self.classifier(x)
        return x

class Resnet(nn.Module):
    def __init__(self, num_classes=101, hidden_size=512, num_layers=1, bidirectional=True):
        super(Resnet, self).__init__()

        # encoder(resnet18)
        model = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        
        # lstm
        self.lstm = nn.LSTM(model.fc.in_features, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # classifier
        if bidirectional:
            self.classifier = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
        self.classifier.apply(initialize_linear)

    def forward(self, x):
        b, d, c, h, w = x.shape
        x = self.encoder(x.view(b * d, c, h, w))
        x = self.lstm(x.view(b, d, -1))[0][:, -1]
        x = self.classifier(x)
        return x