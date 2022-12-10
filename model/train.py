import torch
import torch.nn as nn
import os
import numpy as np
from collections import defaultdict
from torchvision import datasets, models, transforms
from tqdm import tqdm
import argparse

def train_model(model, data_loader, loss_fn, optimizer, scheduler, device):
    model = model.train()

    losses = []
    correct_predictions = 0.
    c = 0

    for d in tqdm(data_loader):  
        # image = d[0].to(device)
        # targets = d[1].to(device)
        image = d[0].to(device)
        targets = d[1].to(device)

        optimizer.zero_grad()

        # get batch_size
        if c == 0:
            batch_size = image.shape[0]  
            c += 1

        outputs = model( 
          image
        ) 
        _, preds = torch.max(outputs, dim=1)  
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
         
        loss.backward() 
        optimizer.step() 
        # scheduler.step()
     
    n_examples = len(data_loader) * batch_size 
    acc = correct_predictions.double() / n_examples 
    return model, acc, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()

    losses = []
    correct_predictions = 0.
    c = 0

    with torch.no_grad():
        for d in tqdm(data_loader): 
            image = d[0].to(device)
            targets = d[1].to(device)  
            if c == 0:
                batch_size = image.shape[0]  
                c += 1

            outputs = model( 
              image
            ) 
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    n_examples = len(data_loader) * batch_size 
    acc = correct_predictions.double() / n_examples 
    return model, acc, np.mean(losses)

if __name__ == '__main__': 
    ###############################
    # Configuration - you can edit
    ############################### 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--batch_size', type=int, default=10) 
    parser.add_argument('--feature_dim', type=int, default=512) 
    parser.add_argument('--class_num', type=int, default=8)  
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='./data/FI')
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    args = parser.parse_args()  

    batch_size = args.batch_size 
    feature_dim = args.feature_dim 
    class_num = args.class_num 
    learning_rate = args.lr  
    num_epoch = args.num_epoch  
    data_dir = args.data_dir 
    save_dir = args.save_dir    
    device = torch.device('cuda:0')   

    ###############################
    # Load data
    ############################### 
    input_size = 256
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    } 
    # Create train/val/test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val', 'test']}

    train_data_loader = dataloaders_dict['train']
    val_data_loader = dataloaders_dict['val']
    test_data_loader = dataloaders_dict['test']

    ###############################
    # Define a model and optimizer
    ###############################
    resnet18 = models.resnet18(pretrained = True)
    resnet18.fc = nn.Linear(feature_dim, class_num)
    resnet18 = resnet18.to(device) 

    optimizer = torch.optim.Adam(resnet18.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.96) 
    loss_fn = nn.CrossEntropyLoss().to(device)

    ###############################
    # Train
    ###############################
    total_steps = len(train_data_loader) * num_epoch
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(num_epoch):

        print(f'Epoch {epoch + 1}/{num_epoch}')
        print('-' * 10)

        resnet18, train_acc, train_loss = train_model(resnet18, train_data_loader, loss_fn, optimizer, scheduler, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        resnet18, val_acc, val_loss = eval_model(resnet18, val_data_loader, loss_fn, device)

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if best_accuracy < val_acc:
            best_accuracy = val_acc
            # save model parameters
            filename = os.path.join(save_dir, 'resnet18.pt')
            torch.save(resnet18.state_dict(), filename)
        
    resnet18, test_acc, test_loss = eval_model(resnet18, test_data_loader, loss_fn, device)
    print(f'Learning rate : {learning_rate:10}')
    print(f'Test acc : {test_acc:10}')

