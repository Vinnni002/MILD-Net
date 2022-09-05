from dataset import Aug_GlaS
from Model.model import MILD_Net, UNet_Contour
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import torch
import pickle
from utils import eval, dice_loss
import matplotlib.pyplot as plt

# Parse cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bs")
parser.add_argument("--ep")
parser.add_argument("--device")
args = parser.parse_args()
devices = args.device.split('-')
devices = [int(i) for i in devices]

# Creating object of dataset class
dataset = Aug_GlaS(transform = True)
train_len = int(dataset.__len__() * 0.8)
val_len = dataset.__len__() - train_len
# testA = Aug_GlaS(testA=True)
# testB = Aug_GlaS(testB=True)
# val = torch.utils.data.ConcatDataset([testA, testB])
train, val = torch.utils.data.random_split(dataset, [train_len, val_len])

# print(dataset.__len__())
# print(train_len)
# print(val_len)

# Defining DataLoader
bs = int(args.bs)
train_dl = DataLoader(train, shuffle = True, batch_size = bs)
val_dl = DataLoader(val, shuffle = True, batch_size = bs)


# Initialization
device = torch.device('cuda:' + str(devices[0]))
print('Using GPU device :', devices)
model = MILD_Net(3, 1)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
# opt = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.99, weight_decay = 0.0005)
opt = torch.optim.Adam(model.parameters(), lr = 10e-4)
model = torch.nn.DataParallel(model, device_ids = devices[0:2])
# model.load_state_dict(torch.load('Weights/epochs/epoch_50', map_location = device))
model = model.to(device)

device_1 = torch.device('cuda:' + str(devices[2]))
unet = UNet_Contour(4, 3)
unet = torch.nn.DataParallel(unet, device_ids = devices[2:4])
unet = unet.to(device_1)

# Training
epochs = int(args.ep)
max_val = 0
min_val = 10e4
train_history = []
val_history = []
train_metric = []
val_metric = []

# with open('history/history/train_history', 'rb') as f:
#     train_history = pickle.load(f)
# with open('history/history/val_history', 'rb') as f:
#     val_history = pickle.load(f)
# with open('history/history/train_metric', 'rb') as f:
#     train_metric = pickle.load(f)
# with open('history/history/val_metric', 'rb') as f:
#     val_metric = pickle.load(f)
    
for epoch in range(epochs):
    total = 0
    total_met = 0
    for (i, (img, img_seg, img_cont, img_anno)) in tqdm.tqdm(enumerate(train_dl), "Epoch : {}".format(epoch + 1)):
        img = img.to(torch.float).to(device)
        ps, pc = model(img)
        ps = ps.squeeze(1)
        pc = pc.squeeze(1)
        img_seg = img_seg.to(torch.float).to(device)
        img_cont = img_cont.to(torch.float).to(device)

        loss1 = criterion(ps, img_seg)
        loss2 = criterion(pc, img_cont)
        loss = loss1 + loss2 

        ps_th = torch.where(ps > 0.5, 1, 0)
        dse_gt = torch.ones(ps_th.shape)
        dse_gt[torch.where(ps_th == img_seg)] = 0
        dse_gt[torch.where(ps_th > img_seg)] = 2
        dse_gt = F.one_hot(dse_gt.long(), num_classes = 3).permute(0, 3, 1, 2)
        ps_th = ps_th.unsqueeze(1)
        # print(img.shape, ps_th.shape)
        dse_ip = torch.concat((img, ps_th), dim = 1)
        dse_op = unet(dse_ip)

        dse_loss = dice_loss(dse_op, dse_gt.to(device_1)).to(device)
        loss += dse_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.detach().cpu()
        
        ps = torch.where(ps > 0.5, 1, 0)
        pc = torch.where(pc > 0.5, 0, 1)
        res = ps.logical_and(pc)
        

    # total_met = eval(model, train, device)
    total_val = 0

    for (i, (img, img_seg, img_cont, img_anno)) in tqdm.tqdm(enumerate(val_dl), "Epoch : {}".format(epoch + 1)):
        img = img.to(torch.float).to(device)
        ps, pc = model(img)
        ps = ps.squeeze(1)
        pc = pc.squeeze(1)
        img_seg = img_seg.to(torch.float).to(device)
        img_cont = img_cont.to(torch.float).to(device)

        loss1 = criterion(ps, img_seg)
        loss2 = criterion(pc, img_cont)
        loss = loss1 + loss2 

        ps_th = torch.where(ps > 0.5, 1, 0)
        dse_gt = torch.ones(ps_th.shape)
        dse_gt[torch.where(ps_th == img_seg)] = 0
        dse_gt[torch.where(ps_th > img_seg)] = 2
        dse_gt = F.one_hot(dse_gt.long(), num_classes = 3).permute(0, 3, 1, 2)
        ps_th = ps_th.unsqueeze(1)
        # print(img.shape, ps_th.shape)
        dse_ip = torch.concat((img, ps_th), dim = 1)
        dse_op = unet(dse_ip)

        dse_loss = dice_loss(dse_op, dse_gt.to(device_1)).to(device)
        loss += dse_loss       
        total_val += loss.detach().cpu()
    
    # total_val_met = eval(model, val, device)
    total_val_met = 0
    
    total = total / (train.__len__() / bs)
    total_val = total_val / (val.__len__() / bs)

    # if(epoch == 100):
    #     opt = torch.optim.Adam(model.parameters(), lr = 10e-5)

    # if total_val_met > max_val:
    #     max_val = total_val_met
    #     torch.save(model.state_dict(), 'Weights/best_score')

    if ((epoch + 1) % 1 == 0):
#         # A = eval(model, testA, device)
#         # B = eval(model, testB, device)
#         # C = (A * testA.__len__() + B * testB.__len__()) / (testA.__len__() + testB.__len__())
#         # torch.save(model.state_dict(), 'Weights/epoch_' + str(epoch + 1) + '_' + str(A) + '_' + str(B) + '_' + str())
        torch.save(model.state_dict(), 'Weights/dse_epochs/model_epoch_' + str(epoch + 1))
        torch.save(unet.state_dict(), 'Weights/dse_epochs/unet_epoch_' + str(epoch + 1))

    # if ((epoch + 1) % 10 == 0):
    #     # A = eval(model, testA, device)
    #     # B = eval(model, testB, device)
    #     # C = (A * testA.__len__() + B * testB.__len__()) / (testA.__len__() + testB.__len__())
    #     # print(A, B, C)
    #     met = eval(model, val, device)
    #     torch.save(model.state_dict(), 'Weights/dse_epoch_' + str(epoch + 1) + '_' + str(met))

    # if total_val < min_val:
    #     min_val = total_val
    #     torch.save(model.state_dict(), 'Weights/best_loss')
    
    print('Total Loss : {}, F1 Score Training : {}, Total Val Loss : {}, F1 Score Validation : {}'.format(total, total_met, total_val, total_val_met))
    train_history.append(total)
    val_history.append(total_val)
#     train_metric.append(total_met)
#     val_metric.append(total_val_met)
    # lr = (1 - ((epoch + 1) / epochs)) ** 2 * lr
    # opt = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.99, weight_decay = 0.0005)

torch.save(model.state_dict(), 'Weights/last_epoch')

with open("history/train_history", "wb") as f:
    pickle.dump(train_history, f)

with open("history/val_history", "wb") as f:
    pickle.dump(val_history, f)

# with open("history/train_metric", "wb") as f:
#     pickle.dump(train_metric, f)

# with open("history/val_metric", "wb") as f:
#     pickle.dump(val_metric, f)