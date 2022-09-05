import torch
import glob
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.functional as F

class GlaS(Dataset):
    def __init__(self, train = True, testA = False, testB = False, transform = None):
        self.transform = transform
        if testA:
            self.name = 'testA'
        elif testB:
            self.name = 'testB'
        else:
            self.name = 'train'

    def __len__(self):
        q = 3
        list = glob.glob('../GlaS/' + self.name + '*')
        return int(len(list) / q)

    def __getitem__(self, idx):
        a = 512
        img = torch.from_numpy(cv2.resize(cv2.imread('../GlaS/' + self.name + '_' + str(idx + 1) + '.bmp'), (a, a), interpolation = cv2.INTER_NEAREST))
        # img_cont = torch.from_numpy(cv2.resize(cv2.imread('../GlaS/' + self.name + '_' + str(idx + 1) + '_cont.png'), (a, a), interpolation = cv2.INTER_NEAREST))
        # img_anno_1 = torch.from_numpy(cv2.resize(cv2.imread('../GlaS/' + self.name + '_' + str(idx + 1) + '_anno.bmp'), (a, a), interpolation = cv2.INTER_NEAREST))
        img_cont = torch.from_numpy(cv2.resize(cv2.imread('../GlaS/distorted/' + self.name + '_' + str(idx + 1) + '_cont.png'), (a, a), interpolation = cv2.INTER_NEAREST))
        img_anno = torch.from_numpy(cv2.resize(cv2.imread('../GlaS/distorted/' + self.name + '_' + str(idx + 1) + '_anno.bmp'), (a, a), interpolation = cv2.INTER_NEAREST))

        img = img.permute(2, 0, 1).to(torch.float)
        img_cont = img_cont.permute(2, 0, 1)
        img_anno = img_anno.permute(2, 0, 1)
        # img_anno_1 = img_anno_1.permute(2, 0, 1)
        
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        if self.transform:
            # crop = 464
            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop, crop))
            # img = TF.crop(img, i, j, h, w)
            # img_cont = TF.crop(img_cont, i, j, h, w)
            # img_anno = TF.crop(img_anno, i, j, h, w)
            
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_cont = TF.hflip(img_cont)
                img_anno = TF.hflip(img_anno)
                # img_anno_1 = TF.hflip(img_anno_1)

            if random.random() > 0.5:
                img = TF.vflip(img)
                img_cont = TF.vflip(img_cont)
                img_anno = TF.vflip(img_anno)
                # img_anno_1 = TF.vflip(img_anno_1)

            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            img_cont = TF.rotate(img_cont, angle)
            img_anno = TF.rotate(img_anno, angle)
            # img_anno_1 = TF.rotate(img_anno_1, angle)

            r_d = random.uniform(0, 0.1)
            g_d = random.uniform(0, 0.1)
            b_d = random.uniform(0, 0.1)

            img[0, :, :] += r_d
            img[1, :, :] += g_d
            img[2, :, :] += b_d
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

            # sig = random.uniform(0, 2)
            # gb = transforms.GaussianBlur(kernel_size = 3, sigma = sig)
            # img = gb(img)
        
        img_seg = torch.where(img_anno == 0, 0, 1)[0, :, :]
        img_cont = img_cont[0, :, :]
        img_anno = img_anno[0, :, :]
        # img_anno_1 = img_anno_1[0, :, :]
        
        return img, img_seg, img_cont

class GlaSD(Dataset):
    def __init__(self, img_dir, transform = None):
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return int(len(glob.glob(self.img_dir + "/*")) / 4)
    
    def __getitem__(self, idx):
        images = glob.glob(self.img_dir + '/*anno.bmp')
        name = images[idx].split('/')[-1][:-9]
        img = cv2.imread(self.img_dir + '/' + name +'.bmp')
        img_seg = np.squeeze(np.where(np.delete(np.array(cv2.imread(self.img_dir + '/' +name +'_anno.bmp')), [1, 2], axis = 2) == 0, 0, 1), axis = 2)
        img_cont = np.squeeze(np.where(np.delete(np.array(cv2.imread(self.img_dir + '/' +name +'_contours.png')), [1, 2], axis = 2) == 0, 0, 1), axis = 2)
        
        a = 512
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        img_seg = torch.from_numpy(img_seg).squeeze(0)
        img_cont = torch.from_numpy(img_cont).squeeze(0)
        img_inst = torch.from_numpy(np.squeeze(np.delete(np.array(cv2.imread(self.img_dir + '/' + name +'_anno.bmp')),[1, 2], axis = 2)))
        
        if self.transform:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(400, 400))
            img = TF.crop(img, i, j, h, w)
            img_seg = TF.crop(img_seg, i, j, h, w)
            img_cont = TF.crop(img_cont, i, j, h, w)
            img_inst = TF.crop(img_inst, i, j, h, w)
            
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_seg = TF.hflip(img_seg)
                img_cont = TF.hflip(img_cont)
                img_inst = TF.hflip(img_inst)

            if random.random() > 0.5:
                img = TF.vflip(img)
                img_seg = TF.vflip(img_seg)
                img_cont = TF.vflip(img_cont)
                img_inst = TF.vflip(img_inst)
        
        
        img = TF.resize(img, size=[a, a])
        img_seg = TF.resize(img_seg.unsqueeze(0), size=[a, a])
        img_cont = TF.resize(img_cont.unsqueeze(0), size=[a, a])
        img_inst = TF.resize(img_inst.unsqueeze(0), size=[a, a])
        
        return img, img_seg.squeeze(0), img_cont.squeeze(0), torch.rand(1, 3, 512, 512), img_inst.squeeze(0)


class Aug_GlaS(Dataset):
    def __init__(self, train = True, testA = False, testB = False, transform = None):
        self.transform = transform
        if testA:
            self.name = 'testA'
        elif testB:
            self.name = 'testB'
        else:
            self.name = 'train'

    def __len__(self):
        # q = 4
        list = glob.glob('../Aug_GlaS/' + self.name + '*anno*.bmp')
        return int(len(list))

    def __getitem__(self, idx):
        a = 512
        img = torch.from_numpy(cv2.imread('../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '.bmp'))
        img_cont = torch.from_numpy(cv2.imread('../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '_cont.png'))
        img_anno = torch.from_numpy(cv2.imread('../Aug_GlaS/' + self.name + '_' + str(idx + 1) + '_anno.bmp'))

        # if [img.shape[1],img.shape[2]] != [512, 512]:
        #     img = torch.from_numpy(cv2.resize(img.numpy(), (a, a), cv2.INTER_NEAREST))
        #     img_cont = torch.from_numpy(cv2.resize(img_cont.numpy(), (a, a), cv2.INTER_NEAREST))
        #     img_anno = torch.from_numpy(cv2.resize(img_anno.numpy(), (a, a), cv2.INTER_NEAREST))

        img = img.permute(2, 0, 1).to(torch.float)
        img_cont = img_cont.permute(2, 0, 1)
        img_anno = img_anno.permute(2, 0, 1)
        
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        if self.transform:
            # crop = 464
            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop, crop))
            # img = TF.crop(img, i, j, h, w)
            # img_cont = TF.crop(img_cont, i, j, h, w)
            # img_anno = TF.crop(img_anno, i, j, h, w)
            
            if random.random() > 0.5:
                img = TF.hflip(img)
                img_cont = TF.hflip(img_cont)
                img_anno = TF.hflip(img_anno)

            if random.random() > 0.5:
                img = TF.vflip(img)
                img_cont = TF.vflip(img_cont)
                img_anno = TF.vflip(img_anno)

            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            img_cont = TF.rotate(img_cont, angle)
            img_anno = TF.rotate(img_anno, angle)

            r_d = random.uniform(0, 0.1)
            g_d = random.uniform(0, 0.1)
            b_d = random.uniform(0, 0.1)

            img[0, :, :] += r_d
            img[1, :, :] += g_d
            img[2, :, :] += b_d
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

            sig = random.uniform(0, 2)
            gb = transforms.GaussianBlur(kernel_size = 3, sigma = sig)
            img = gb(img)
        
        img_seg = torch.where(img_anno == 0, 0, 1)[0, :, :]
        img_cont = img_cont[0, :, :]
        img_anno = img_anno[0, :, :]
        
        # img_seg = F.one_hot(img_seg.long(), num_classes = 2).permute(2, 0, 1)
        # img_cont = F.one_hot(img_cont.long(), num_classes = 2).permute(2, 0, 1)

        return img, img_seg, img_cont, img_anno