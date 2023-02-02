import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

"""
INPUT: 
    An optional boolean argument (default value is True for training dataset)

RETURNS:
    Dataloader for the training set (if training = True) or the test set (if training = False)
"""
def get_data_loader(training = True):

    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if (training):
        train_set = datasets.FashionMNIST('./data', train = True, download = True, transform = custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    else:
        test_set = datasets.FashionMNIST('./data', train = False, transform = custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
    
    return loader

"""
INPUT: 
    None

RETURNS:
    An untrained neural network model
"""
def build_model():
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 28 * 28,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )

    return model

"""
INPUT: 
    model - the model produced by the previous function
    train_loader  - the train DataLoader produced by the first function
    criterion   - cross-entropy 
    T - number of epochs for training

RETURNS:
    None
"""
def train_model(model, train_loader, criterion, T):
    
    opt = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    model.train()

    for epoch in range(T):
        running_loss = 0
        correct = 0
        total = 0
        accuracy = 0

        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [images, labels]
            images, labels = data
            
            # Zero the parameter gradients
            opt.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(images)
            # outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # Track running loss
            running_loss += loss.item()
            # prediction accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Train Epoch: {epoch}  Accuracy: {correct}/{total}({accuracy:.2f}%)  Loss: {running_loss / len(train_loader):.3f}')

"""
INPUT: 
    model - the the trained model produced by the previous function
    test_loader    - the test DataLoader
    criterion   - cropy-entropy 

RETURNS:
    None
"""
def evaluate_model(model, test_loader, criterion, show_loss = True):
    
    model.eval()

    with torch.no_grad():
        running_loss = 0
        total = 0
        correct = 0

        for data, labels in test_loader:
            # Get the inputs; data is a list of [images, labels]
            input = data

            # Forward + Backward + Optimize
            outputs = model(input)
            # outputs = model(images)
            loss = criterion(outputs, labels)

            # Track running loss
            running_loss += loss.item()
            # prediction accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        if (show_loss):
            print(f'Average loss: {running_loss / len(test_loader):.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
        else:
            print(f'Accuracy: {accuracy:.2f}%')

"""
INPUT: 
    model - the trained model
    test_images   -  test image set of shape Nx1x28x28
    index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


RETURNS:
    None
"""
def predict_label(model, test_images, index):
    classNames = {
        'T-shirt/top' : 0,
        'Trouser' : 0,
        'Pullover' : 0,
        'Dress' : 0,
        'Coat' : 0,
        'Sandal' : 0,
        'Shirt' : 0,
        'Sneaker' : 0,
        'Bag' : 0,
        'Ankle Boot' : 0
    }
    image = test_images[index]
    output = model(image)
    
    prob = F.softmax(output, dim = 1)

    # Getting top three
    probNP = []
    for index in prob[0]:
        probNP.append(index.item())

    x = 0
    for key in classNames:
        classNames[key] = probNP[x]
        x += 1

    sortedClassNames = sorted(classNames.items(), key=lambda x:x[1], reverse=True)
    print(f'{sortedClassNames[0][0]}: {sortedClassNames[0][1] * 100 :.2f}%')
    print(f'{sortedClassNames[1][0]}: {sortedClassNames[1][1] * 100 :.2f}%')
    print(f'{sortedClassNames[2][0]}: {sortedClassNames[2][1] * 100 :.2f}%')
    
    

if __name__ == '__main__':
    # get_data_loader
    print("Train Loader")
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)

    print("\nTest Loader")
    test_loader= get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)

    # build_model
    print("\nBuild Model")
    model = build_model()
    print(model)

    # train_model
    print("\nTrain Model")
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)

    # evaluate_model
    print("\nEvaluate Model")
    print("  Loss = False:")
    evaluate_model(model, test_loader, criterion, False)
    print("  Loss = True:")
    evaluate_model(model, test_loader, criterion, True)

    # predict_model
    print("\nPredict Model")
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)