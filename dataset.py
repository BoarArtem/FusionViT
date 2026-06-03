from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

train_transforms = get_train_transforms()
test_transforms = get_test_transforms()

def get_train_dataset():
    return datasets.CIFAR100(root="data", train=True, transform=train_transforms, download=True)

def get_test_dataset():
    return datasets.CIFAR100(root="data", transform=test_transforms, train=False, download=False)

train_dataset = get_train_dataset()
test_dataset = get_test_dataset()

def get_train_loader():
    return DataLoader(train_dataset, batch_size=128, shuffle=True)

def get_test_loader():
    return DataLoader(test_dataset, batch_size=128, shuffle=False)

train_loader = get_train_loader()
test_loader = get_test_loader()

# for inputs, labels in train_loader:
#     print(inputs.shape)
#     print(labels.shape)
#
#     break