import re
import shutil
import librosa.display
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from collections import Counter
from torch.utils.data import Subset
import torchaudio
from torch.utils.data import DataLoader
from torchaudio import transforms
import numpy as np
import torchvision
import sys
import os
import torch.nn as nn
#sys.path.insert(1, '../script')

from model import EfficientNetNotRGB
from loader import SoundDS
from AudioUtil import AudioUtil
from SAM import SAM
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import time, copy, argparse
from tqdm import tqdm
import wandb

from torch.utils.data import DataLoader, Dataset, random_split
#from efficientnet_pytorch import EfficientNet'

def train_model(model, dataloaders, train_directory, criterion, optimizer, name, num_epochs=25):
    since = time.time()
    
    val_acc_history = []
    names = os.listdir('../data/'+train_directory)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999999.9
    model = model.to(device)
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device,dtype=torch.float)
                labels = labels.to(device)
                if len(labels) != 1:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        # statistics
                        running_corrects += torch.sum(preds == labels.data)
                        del preds, outputs
                        if phase == 'train':
                            def closure():
                                loss = criterion(model(inputs), labels)
                                loss.backward()
                                return loss
                            loss = criterion(model(inputs), labels)
                            loss.backward()
                            optimizer.step(closure)
                            optimizer.zero_grad()
                        running_loss += loss.item() * inputs.size(0)
            if len(labels) == 1:
                epoch_loss = running_loss / (len(dataloaders[phase].dataset)-1)
                epoch_acc = float(running_corrects) / (len(dataloaders[phase].dataset)-1)
            else:
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            wandb.log({"Loss_"+phase: epoch_loss,"Accuracy_"+phase: epoch_acc})
            # deep copy the model
            if phase == 'val': #and (epoch_acc >= best_acc or best_loss > epoch_loss):
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                if best_loss > epoch_loss:
                    best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
       
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, '../models/'+name+'/'+name+'val_acc_'+str(best_acc)+'val_loss_'+str(best_loss))
                wandb.save(name)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #TODO RUN TEST SET Accuracy
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    phase = 'test'
    total_cf_matrix = np.zeros((len(names),len(names)))
    total_labels = Counter([])
    total_preds = Counter([])
    for inputs, labels in dataloaders['test']:
        if len(labels) != 1:
            inputs = inputs.to(device,dtype=torch.float)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total_labels = total_labels + Counter(labels.tolist())
                total_preds = total_preds + Counter(preds.tolist())
                cf_matrix = confusion_matrix(preds.tolist(),labels.tolist(),labels=range(len(names)))
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_cf_matrix = total_cf_matrix + cf_matrix
            #print(cf_matrix)
    print("test set loss and acc :: ",epoch_loss, epoch_acc)
    print('number of total tests ')
    print(str(total_labels))
    print('prediction counts')
    print(str(total_preds))
    print('confusion matrix')
    print(str(total_cf_matrix))
    if len(labels) == 1:
        epoch_loss = running_loss / (len(dataloaders[phase].dataset)-1)
        epoch_acc = float(running_corrects) / (len(dataloaders[phase].dataset)-1)
    else:
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)



if __name__ == "__main__":
    #parameters
    model_save_name = 'dog_audio_effB1_ep120'
    batch_size = 16
    ep = 120
    lr = 0.01
    num_classes = 9
    train_directory = 'dog'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dt = SoundDS(data_path='../data/'+train_directory, transform=use_transform)
    #for graphs
    wandb.init(project=model_save_name, entity='leejaeka', save_code=True)
    config = wandb.config
    config.learning_rate = 0.01

    train_len = int(0.8*len(dt))
    val_len = int(0.1*len(dt))
    test_len = len(dt) - train_len - val_len
    train_dt, val_dt, test_dt = torch.utils.data.random_split(dt, [train_len,val_len,test_len],generator=torch.Generator().manual_seed(27))
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_size, shuffle=False)
    print(len(train_dt), 'TRAIN LEN')
    print(len(val_dt), 'VAL LEN')
    print(len(test_dt), 'TEST LEN')

    # initialize model
    deepNet = EfficientNetNotRGB.from_pretrained('efficientnet-b1',num_classes=num_classes)
    # optimizer
    params_to_update = deepNet.parameters()
    optimizer_ft = optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss()
    #create save folder
    if not os.path.exists('../models/'+model_save_name):
        os.makedirs('../models/'+model_save_name)
    deepNet, hist = train_model(deepNet, {'train':train_loader, 'val':val_loader, 'test':test_loader}, train_directory, criterion, optimizer_ft, model_save_name, num_epochs=ep)    



















