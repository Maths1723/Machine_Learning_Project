# - - - Libs - - -
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# - - - Networks - - -
class MLP( nn.Module ):
    def __init__( self, num_classes ):
        super().__init__()
        # Multi Layer Perceptron
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 100*100*3, 128 ),
            nn.ReLU(),
            nn.Linear( 128, 64 ),       
            nn.ReLU(),
            nn.Linear(64,num_classes))
    def forward( self, x ):
        return self.layers( x )

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Convolutional Neural Network Layer 
        self.convolutional_neural_network_layers = nn.Sequential( # (N, 3, 100, 100)
                nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=1), 
                nn.ReLU(), # (N, 12, 100, 100)
                nn.MaxPool2d(kernel_size=2), # (N, 12, 50, 50)
                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(), # (N, 24, 50, 50)
                nn.MaxPool2d(kernel_size=2), # (N, 24, 25, 25)
        )
        # Linear layer
        self.linear_layers = nn.Sequential(
                nn.Linear(in_features=24 * 25 * 25, out_features=64),          
                nn.ReLU(),
                # Dropout with probability of 0.2 to avoid overfitting :
                nn.Dropout(p=0.2),
                nn.Linear(in_features=64, out_features= num_classes)
        )
    # Defining the forward pass 
    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# - - - Saving - - -
def save_model_MLP(model, path = "tomato_MLP.pth"):
    """Save the model's state dictionary to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_model_CNN(model, path = "tomato_CNN.pth"):
    """Save the model's state dictionary to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# - - - Training - - -
def train( data_loader, model, criterion, optimizer, device ):
    # Number of batches :
    num_batches = len( data_loader )
    # Total number of samples :
    num_items = len( data_loader.dataset )
    total_loss = 0
    
    for data, target in data_loader :
        data = data.to( device )
        target = target.to( device )
        # Forward pass :
        output = model( data )
        loss = criterion( output, target )
        total_loss += loss
        # Backpropagation :
        loss.backward()
        # Update the weights :
        optimizer.step()
        # Cancel the gradient :
        optimizer.zero_grad()

    train_loss = total_loss / num_batches
    print( f"average loss {train_loss:7f}" )

# - - - Testing - - -
def test( test_loader, model, criterion, device ) :
    model.eval()
    # Number of batches :
    num_batches = len( test_loader )
    test_loss = 0
    
    with torch.no_grad() :
        for data, target in test_loader :
            data = data.to( device )
            target = target.to( device )
            # Forward pass with computed weights :
            output = model( data )
            loss = criterion( output, target )
            test_loss += loss.item()
            
    test_loss = test_loss / num_batches
    print( f"average loss {test_loss:7f}" )

# - - - Plotting - - -
def visualize_errors(test_loader, model, criterion, max_checks, max_errors, device):
    model.eval()
    num_checks = 0
    num_errors = 0
    error_images = []
    error_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, dim=1)
            
            for i in range(data.size(0)):
                if num_checks >= max_checks or num_errors >= max_errors:
                    break
                num_checks += 1
                if predicted[i].item() != target[i].item():
                    num_errors += 1
                    true_prob = probabilities[i, target[i].item()].item()
                    max_prob = max_probs[i].item()
                    error_images.append(data[i].cpu())
                    error_labels.append((target[i].item(), true_prob, predicted[i].item(), max_prob))
                
                if num_checks >= max_checks or num_errors >= max_errors:
                    break
            
            if num_checks >= max_checks or num_errors >= max_errors:
                break
    
    if num_errors > 0:
        cols = min(num_errors, 3)
        rows = (num_errors + cols - 1) // cols
        plt.figure(figsize=(cols * 2, rows * 2))
        
        for i, (image, (true_label, true_prob, pred_label, max_prob)) in enumerate(zip(error_images, error_labels)):
            plt.subplot(rows, cols, i + 1)
            
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            image_display = np.transpose(image, (1, 2, 0))
            
            plt.imshow(image_display)
            plt.title(f"{true_label}:({true_prob:.3f}), {pred_label}:({max_prob:.3f})")
            plt.axis('off')
        
        plt.tight_layout()
        if isinstance(model,MLP):
            plt.savefig("tomato_errors_MLP.png")
        elif isinstance(model,CNN):
            plt.savefig("tomato_errors_CNN.png")
        plt.show()
        plt.close()
        print("Error images saved")
    else:
        print("No errors found within the checked samples.")
    accuracy=(1-num_errors/num_checks)*100
    print(f"Checked: {num_checks}, Errors: {num_errors}, Accuracy: {accuracy:.3f}")

# - - - System - - -
def get_device():
    if torch.cuda.is_available():
        print( "using GPU, device name:", torch.cuda.get_device_name( 0 ) )
        device = torch.device( 'cuda' )
    else:
        print( "no GPU found, using CPU" )
        device = torch.device( 'cpu' )
    return device

# - - - User - - -
def get_user_choice():
    while True:
        choice = input("Do you want to (t)rain a new model or (l)oad a saved model? [t/l]: ").lower()
        if choice in ['t', 'l']:
            return choice
        print("Invalid input. Please enter 't' or 'l'.")
