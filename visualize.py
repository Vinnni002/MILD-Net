from utils import visualize
import argparse
from Model.model import MILD_Net
import torch
from dataset import Aug_GlaS

train = Aug_GlaS(transform=True)
testA = Aug_GlaS(testA=True)
testB = Aug_GlaS(testB=True)
val = torch.utils.data.ConcatDataset([testA, testB])

parser = argparse.ArgumentParser()
parser.add_argument('--ds')
parser.add_argument('--start')
parser.add_argument('--fn')
parser.add_argument('--ms')
parser.add_argument('--device')
args = parser.parse_args()

device = torch.device('cuda:' + args.device)
devices = args.device.split('-')
devices = [int(i) for i in devices]

model = MILD_Net(3, 1)
model = torch.nn.DataParallel(model, device_ids = devices)
model.load_state_dict(torch.load('Weights/' + args.ms, map_location = device))
model.to(device)

if args.ds == 'train':
    ds = train
elif args.ds == 'testA':
    ds = testA
elif args.ds == 'testB':
    ds = testB
elif args.ds == 'val':
    ds = val

for i in range(0, len(ds), 5):
    print(args.fn + str(i) + '.png')
    visualize(model, i, ds, fn = args.fn + str(i) + '.png')

# visualize(model, int(args.start), ds, fn = args.fn)