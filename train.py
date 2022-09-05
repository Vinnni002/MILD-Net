from dataset import Aug_GlaS as Aug_GlaS
from Model.model import MILD_Net
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch
import pickle
from utils import eval, dice_loss, WeightedFocalLoss
import torch.nn.functional as F

# Parse cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bs")
parser.add_argument("--ep")
parser.add_argument("--device")
args = parser.parse_args()
devices = args.device.split('-')
devices = [int(i) for i in devices]

# Creating object of dataset class
train = Aug_GlaS(transform = True)
# train_len = int(train.__len__() * 0.8)
# val_len = train.__len__() - train_len
testA = Aug_GlaS(testA=True)
testB = Aug_GlaS(testB=True)
val = torch.utils.data.ConcatDataset([testA, testB])
# train, val = torch.utils.data.random_split(train, [train_len, val_len])

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
# criterion = WeightedFocalLoss(device=device)
# opt = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.99, weight_decay = 0.0005)
opt = torch.optim.Adam(model.parameters(), lr = 10e-4, weight_decay = 10e-5)
model = torch.nn.DataParallel(model, device_ids = devices)
# model.load_state_dict(torch.load('Weights/attn_bce/epoch_145', map_location = device))
model = model.to(device)

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
        # img_seg = F.one_hot(img_seg.long(), num_classes = 2).permute(0, 3, 1, 2)
        # img_cont = F.one_hot(img_cont.long(), num_classes = 2).permute(0, 3, 1, 2)
        # loss3 = dice_loss(img_seg, ps)
        # loss4 = dice_loss(img_cont, pc)
        loss = loss1 + loss2 
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.detach().cpu()
        
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
        # img_seg = F.one_hot(img_seg.long(), num_classes = 2).permute(0, 3, 1, 2)
        # img_cont = F.one_hot(img_cont.long(), num_classes = 2).permute(0, 3, 1, 2)
        # loss3 = dice_loss(img_seg, ps)
        # loss4 = dice_loss(img_cont, pc)
        loss = loss1 + loss2
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
        torch.save(model.state_dict(), 'Weights/attn_bce/epoch_' + str(epoch + 1))

    if ((epoch + 1) % 5 == 0):
    #     # A = eval(model, testA, device)
    #     # B = eval(model, testB, device)
    #     # C = (A * testA.__len__() + B * testB.__len__()) / (testA.__len__() + testB.__len__())
    #     # print(A, B, C)
        met = eval(model, val, device)
        torch.save(model.state_dict(), 'Weights/attn_bce/metric/epoch_' + str(epoch + 1) + '_' + str(met))

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

with open("history/train_history", "wb") as f:
    pickle.dump(train_history, f)

with open("history/val_history", "wb") as f:
    pickle.dump(val_history, f)

# with open("history/train_metric", "wb") as f:
#     pickle.dump(train_metric, f)

# with open("history/val_metric", "wb") as f:
#     pickle.dump(val_metric, f)