import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


project_root = "/Users/jhroe/Documents/transformer/"
training_data_path = os.path.join(project_root, "data")
test_data_path = os.path.join(project_root, "test_data")

training_data = datasets.FashionMNIST(
    root=training_data_path,
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root=test_data_path,
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



