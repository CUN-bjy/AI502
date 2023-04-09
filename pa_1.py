import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm.auto import tqdm
import dcargs

# Set the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import dataclasses
@dataclasses.dataclass
class TrainConfig:
    # general parameters
    exp_name: str = "exp_default"
    mode: str = "FCN"
    
    num_layer: int = 3
    hidden_size: int = 512
    active_fn: str = "sigmoid"
    
    num_layer_conv: int = 3
    hidden_size_conv: int = 5
    active_fn_conv: str = "sigmoid"
    
    optimizer: str = "adam"
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 128
    weight_decay = 0
    
    
train_config = dcargs.parse(TrainConfig, description=__doc__)

# Define the FCN model
class FCN(nn.Module):
    def __init__(self, in_dim = 3072, out_dim = 10):
        super(FCN, self).__init__()
        """
        Fully Connected Network
        """
        self.num_layer = train_config.num_layer
        self.hidden_size = train_config.hidden_size
        self.active_fn = nn.functional.sigmoid if train_config.active_fn == "sigmoid" \
                        else nn.functional.relu if train_config.active_fn == "relu" \
                        else nn.functional.tanh if train_config.active_fn == "tanh" else None
        
        self.layers = []
        for i in range(self.num_layer):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, self.hidden_size, device=device))
                self.add_module("input_layer", self.layers[-1])
            elif i == self.num_layer-1:
                self.layers.append(nn.Linear(self.hidden_size, out_dim, device=device))
                self.add_module("hidden_layer"+str(i-1), self.layers[-1])
            else:
                self.layers.append(nn.Linear(self.hidden_size, self.hidden_size, device=device))
                self.add_module("out_layer", self.layers[-1])

    def forward(self, x):
        """
        Forward function of FCN
        """
        x = x.view(x.size(0), -1)
        for i in range(self.num_layer):
            x = self.active_fn(self.layers[i](x))

        # out = nn.functional.softmax(self.layers[-1](x), dim=-1)
        out = x
        
        return out
    
# Define the CNN model
class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0):
        super(CNN, self).__init__()
        """
        Convolutional Neural Network
        """  
        self.num_layer = train_config.num_layer_conv
        self.hidden_size = train_config.hidden_size_conv
        self.active_fn = nn.functional.sigmoid if train_config.active_fn_conv == "sigmoid" \
                        else nn.functional.relu if train_config.active_fn_conv == "relu" \
                        else nn.functional.tanh if train_config.active_fn_conv == "tanh" else None

        self.conv_layers = []
        for i in range(self.num_layer):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, device=device))
            elif i == self.num_layer-1:
                self.conv_layers.append(nn.Conv2d(in_channels=self.hidden_size, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, device=device))
            else:
                self.conv_layers.append(nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, device=device))
        
        in_dim = 32
        for i in range(self.num_layer): # num_layer
          in_dim = int((in_dim + 2*padding - (kernel_size-1) - 1) / stride + 1)

        self.fcn = FCN(out_channels * in_dim * in_dim).to(device)


    def forward(self, x):
        """
        Forward function of CNN
        """
        for i in range(self.num_layer):
            x = self.active_fn(self.conv_layers[i](x))
        
        out = self.fcn(x)
        return out
    
    
    
# Set the hyperparameters
"""You can change those values"""
learning_rate = train_config.learning_rate
num_epochs = train_config.num_epochs
batch_size = train_config.batch_size
weight_decay = train_config.weight_decay

# Initialize the model and optimizer
model = FCN().to(device) if train_config.mode == "FCN" else CNN().to(device)

"""You can change the optimizer"""
optim_module =  optim.Adam if train_config.optimizer == "adam"\
                else optim.RMSprop if train_config.optimizer == "rmsprop"\
                else optim.SGD if train_config.optimizer == "sgd" else None
optimizer = optim_module(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

# Define the loss function
criterion = nn.CrossEntropyLoss()


# CIFAR preprocessing 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

# Create the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_losses = []
test_accs = []
for epoch in tqdm(range(num_epochs)):
    # Training
    epoch_loss = 0.0
    for images, labels in train_loader:
        # Move the images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Evaluation
    if (epoch + 1) % 1 == 0:
        num_correct = 0
        num_total = 0
        for images, labels in test_loader:
            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            num_correct += torch.sum(torch.argmax(outputs, dim=-1) == labels)
            num_total += len(labels)
        test_accs.append(float(100 * num_correct / num_total))
        print('[epoch %d] test accuracy: %.2f' %
            (epoch + 1, 100 * num_correct / num_total))

    # Print statistics
    train_losses.append(epoch_loss / len(train_loader))
    print('[epoch %d] training loss: %.4f' %
            (epoch + 1, epoch_loss / len(train_loader)))
    
config_dict = train_config.__dict__
log_to_save = dict(config_dict, **{ "loss": train_losses, "acc": test_accs})

import json

with open(f"{train_config.exp_name}.json", "w") as json_file:
    json.dump(log_to_save, json_file, indent=4)

print('Finished Training')