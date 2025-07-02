# - - - Libs - - -
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from FRUTTA_support import MLP, train, test, visualize_errors, save_model_MLP, save_model_CNN, get_device, get_user_choice, CNN


# - - - Parameters - - -
epochs = 10
batch_size = 32

criterion = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam

max_checks= 3200
max_errors= 9

# - - - Path - - -
model_path_MLP = "tomato_MLP.pth"
model_path_CNN = "tomato_CNN.pth"

# - - - GPU handling - - -
device=get_device()

# - - - Set Up Data - - -
data_train = "./tomato_training"
data_test = "./tomato_test"

train_dataset = datasets.ImageFolder( data_train, transform = ToTensor() )
test_dataset = datasets.ImageFolder( data_test, transform = ToTensor() )

num_classes = len(train_dataset.classes)
print(f"Number of classes in the dataset: {num_classes}")

train_loader = DataLoader( dataset = train_dataset, batch_size = batch_size, shuffle = True )
test_loader = DataLoader( dataset = test_dataset, batch_size = batch_size, shuffle = True )

# - - - Model Creation & Training/Loading - - -
model = MLP(num_classes=num_classes).to(device)
optimizer = optimizer_class(model.parameters())

choice = get_user_choice()
if choice == 't':
    # Train new model
    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}")
        train(train_loader, model, criterion, optimizer, device)
    # Save the trained model
    save_model_MLP(model, model_path_MLP)
    
elif choice == 'l':
    # Load saved model
    if os.path.exists(model_path_MLP):
        model.load_state_dict(torch.load(model_path_MLP, map_location=device))
        print(f"Model loaded from {model_path_MLP}")
    else:
        print(f"No saved model found at {model_path_MLP}. Exiting.")
        exit()

# - - - Model Eval on Testset - - -
test( test_loader, model, criterion,device )
# - - - Visualize Errors on Testset - - -
visualize_errors(test_loader, model, criterion, max_checks, max_errors,device)


# - - - Model Creation & Training/Loading - - -
model = CNN(num_classes=num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

choice = get_user_choice()
if choice == 't':
    # Train new model
    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}")
        train(train_loader, model, criterion, optimizer, device)
    # Save the trained model
    save_model_CNN(model, model_path_CNN)
    
elif choice == 'l':
    # Load saved model
    if os.path.exists(model_path_CNN):
        model.load_state_dict(torch.load(model_path_CNN, map_location=device))
        print(f"Model loaded from {model_path_CNN}")
    else:
        print(f"No saved model found at {model_path_CNN}. Exiting.")
        exit()

# - - - Model Eval on Testset - - -
test( test_loader, model, criterion,device )
# - - - Visualize Errors on Testset - - -
visualize_errors(test_loader, model, criterion, max_checks, max_errors,device)
