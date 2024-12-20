from torchvision import models
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

resnet101 = models.resnet101(weights=True)
resnet101.eval()

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ImageFolder(root='dataset', transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for img, _ in dataloader:
    with torch.no_grad():
        out = resnet101(img)

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    img = img[0].numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.suptitle("Predicted class: {}".format(labels[index[0]]))
    plt.title("Confidence: {}%".format(percentage[index[0]].item()))
    plt.tight_layout()
    plt.axis('off')
    plt.show()
