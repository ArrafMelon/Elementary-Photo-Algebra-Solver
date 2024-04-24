import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class_dict = {
    "0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"add":10,"subtract":11,"multiply":12,"divide":13,"point":14,
    "equals":15,"y":16,"z":17
}

class MathSymbolDataset(Dataset):
  def __init__(self, root_dir, transforms=None):
    self.root_dir = root_dir
    self.transforms = transforms
    self.files = os.listdir(root_dir)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.files[index])
    image = Image.open(img_path)

    class_name = img_path.split('/')[-1]
    label_key = class_name.split(' ')[0]
    label = class_dict[label_key]

    if self.transforms:
      for t in self.transforms:
        image = t(image)

    return(image,torch.tensor(label))

class MathModel(nn.Module):
  def __init__(self):
    super(MathModel,self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
    self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
    self.lin = nn.Linear(64 * 37 * 37, 128)
    self.lin2 = nn.Linear(128, 18)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.mp(x)
    x = F.relu(self.conv2(x))
    x = self.mp(x)
    x = F.relu(self.conv3(x))
    x = x.view(-1, 64 * 37 * 37)
    x = self.lin(x)
    x = self.lin2(x)

    return x