from torchvision import models
import torch.nn as nn

model = models.resnext50_32x4d(pretrained=True)

inputs = model.fc.in_features
outputs = 6
clf = nn.Linear(inputs, outputs)


model.fc = clf
