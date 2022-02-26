import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import *
from processing import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from loss import *
from dataset import *
import warnings
warnings.simplefilter("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,target) in enumerate(loop):
        data = data.to(device=device,dtype=torch.float32)
        target = target.to(device=device,dtype=torch.long)
        #forward
        #target = target.permute(0,3,1,2)

    #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions,target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    valid_losses = []
    model = ESC50Model(input_shape=(1,128,431), batch_size=16, num_cats=50).to(device)
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = FocalLoss(gamma=0,logits=True)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    train_loader, val_loader = get_data()
    batch_losses = []
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        trace_y=[]
        trace_yhat=[]
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # checkpoint = {
        #     "state_dict":model.state_dict(),
        #     "optimizer":optimizer.state_dict(),
        #     }
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy()) 
            loss = loss_fn(y_hat, y)     
            batch_losses.append(loss.item())
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        valid_losses.append(batch_losses)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
if __name__ =="__main__":
    main()