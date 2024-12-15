import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.vit import ViT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pathlib import Path

img_path='/Users/anmolchalise/Downloads/tabitha-turner-Lc9czM0IGfU-unsplash.jpg'
path_obj=Path(img_path)
img=Image.open(path_obj)

preprocess = transforms.Compose([
    transforms.Resize(256),             
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img = preprocess(img).unsqueeze(0).to(device)
print(img.shape,'\n')
model = ViT.from_pretrained("google/vit-base-patch16-224").to(device)

with torch.no_grad():
    model.to(device)
    model.eval()
    logits = model(img)

prob=F.softmax(logits, dim=-1)
index = torch.argmax(prob)
print(index)

with open('./Imagenet_labels/imagenet1000_clsidx_to_labels.txt') as f:
    labels = eval(f.read())
    cat = labels.get((index.item()), "Label not found")
    print("The image is of", cat)
