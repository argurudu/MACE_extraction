import numpy as np
import pandas as pd
import os
import sys
import time
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
import densenet_utils

class CXRDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def image_loader(self, image_name):
        image = io.imread(image_name)
        image = (((image - image.min()) / (image.max() - image.min()))*255).astype(np.uint8)
        image = np.stack((image, )*3)
        image = np.transpose(image, (1, 2, 0))
        image = self.augmentations(image)
        return image
    
    def __getitem__(self, index):
        y = self.df.at[self.df.index[index], 'MACE_label'] #Rule-based label, Mistral-generated label, or manually-annotated label
        x = self.image_loader(self.df.at[self.df.index[index], 'image_paths'])
        y = torch.tensor([y], dtype=torch.float)
        return x, y

#Loading pre-divided training, validation, and testing data
train_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((256,256)),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4756, 0.4756, 0.4756], std=[0.3011, 0.3011, 0.3011])
                                    ])
other_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4756, 0.4756, 0.4756], std=[0.3011, 0.3011, 0.3011])
                                    ])
df_train = pd.read_csv('training_data.csv')
df_validation = pd.read_csv('validation_data.csv')
df_test = pd.read_csv('testing_data.csv')
datagen_train = CXRDataset(df =  df_train.copy(), augmentations = train_transform) 
datagen_val = CXRDataset(df = df_validation.copy(), augmentations = other_transform) 
datagen_test = CXRDataset(df = df_test.copy(),augmentations = other_transform) 

#Loading model
model = models.densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(1024, 14),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(14, 1)
)
pretrained_weights_path = "pretrained_weights_path"
state_dict = torch.load(pretrained_weights_path)
model.load_state_dict(state_dict, strict=False)

#Pre-training variables
do_train= True
n_iters = 100
print_every = 1
plot_every = 1
batch_size = 8
model_name = 'densenet121-finetuned'
save_dir = ''
metric_name = 'val_loss'
maximize_metric=False
tensorboard_log_dir = os.path.join(save_dir, 'tensorboard_logs/densenet')
writer = SummaryWriter(log_dir=tensorboard_log_dir)


#Logging & Saving the model
sys.stdout.flush()
save_dir = densenet_utils.get_save_dir(
    save_dir, training=True if do_train else False
)
logger = densenet_utils.get_logger(save_dir, "densenet121-finetuning")
logger.propagate = False
logger.info('do_train: {}, n_iter: {}, batch_size: {}'.format(do_train, n_iters, batch_size))
saver = densenet_utils.CheckpointSaver(
    save_dir=save_dir,
    metric_name=metric_name,
    maximize_metric=maximize_metric,
    log=logger,
)

#Creating dataloaders
train_loader = DataLoader(dataset=datagen_train, shuffle=True, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(dataset=datagen_val,  shuffle=False, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=batch_size, num_workers=4)

#Model Training Loop
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
current_loss = 0
all_losses = []

if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    model = nn.DataParallel(model)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
    
train_losses = []
val_losses = []

if do_train:
    for iter in range(n_iters):
        current_loss=0
        model.train()
        start = time.time()
        print(start)
        
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            if j%100==0:
                print(j, current_loss)
                sys.stdout.flush()
                
            data[0] = data[0].to(device).float()
            data[1] = data[1].to(device)
            
            output = model(data[0])
            target = data[1]
            loss = criterion(output,target)
            
            loss.backward()
            optimizer.step()
            current_loss+=loss.item()

            writer.add_scalar('Train/Loss', loss.item(), iter * len(train_loader) + j)
        
        #Print iteration number, loss, name and guess
        endt = time.time()
        print('iteration: ', iter, current_loss, endt-start)
        train_losses.append(current_loss)

        writer.add_scalar('Train/Average Loss', current_loss/len(train_loader), iter)
        
        if iter % print_every == 0:
            current_loss = 0
            model.eval()
            y_pred = []
            y_true = []
            y_prob= []
            
            for v, data in enumerate(val_loader):
                if v%100==0:
                    print(v)
                    sys.stdout.flush()
                    
                data[0] = data[0].to(device).float()
                data[1] = data[1].to(device)
                
                output = model(data[0])
                target = data[1]
                loss = criterion(output, target)
                
                probs = output.cpu().detach().numpy()
                preds = (torch.sigmoid(torch.tensor(probs)) >=0.5).int().numpy()
                y_pred += list(preds)
                y_prob +=list(probs)
                y_true += list(data[1].cpu().detach().numpy().reshape(-1))
                current_loss+=loss.item()
                
            val_loss_avg = current_loss / len(val_loader)
            val_results = densenet_utils.eval_dict(y=y_true, 
                                                y_pred=y_pred, 
                                                y_prob=y_prob, 
                                                average='macro',
                                                thresh_search=True)
            val_results_str = ", ".join(
                "{}: {:.4f}".format(k, v) for k, v in val_results.items()
            )
            logger.info("VAL - {}, val_loss: {:.4f}".format(val_results_str,val_loss_avg))
            saver.save(
                    iter, model, optimizer, val_loss_avg
                )
            val_losses.append(current_loss)

            writer.add_scalar('Validation/Average Loss', val_loss_avg, iter)
            writer.add_scalar('Validation/AUROC', val_results.get('auroc', 0), iter)
