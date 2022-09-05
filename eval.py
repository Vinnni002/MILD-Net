from Model.model import MILD_Net
from dataset import Aug_GlaS as GlaS
import torch
import argparse
from utils import eval
import matplotlib.pyplot as plt

# Parse cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--ms')
parser.add_argument('--x_th')
parser.add_argument('--y_th')
parser.add_argument('--ds')
args = parser.parse_args()
devices = args.device.split('-')
devices = [int(i) for i in devices]
device = torch.device('cuda:' + str(devices[0]))

testA = GlaS(testA=True)
testB = GlaS(testB=True)
val = torch.utils.data.ConcatDataset([testA, testB])

print('Using cuda device :', devices)

model = MILD_Net(3, 1)
model = torch.nn.DataParallel(model, device_ids = devices)
model.load_state_dict(torch.load('Weights/' + args.ms, map_location = device))
model.to(device)

print('Model Loaded! Running Evaluation....')

if args.ds == 'val':
    ds = val
elif args.ds == 'testA':
    ds = testA
elif args.ds == 'testB':
    ds = testB

score = eval(model, ds, device, float(args.x_th), float(args.y_th))

# print('F1 Score :', score[0], 'Dice Score : ', score[1], 'Hausdorff Distance : ', score[2])
print('F1 Score :', score[0], 'Dice Score : ', score[1])