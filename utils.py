import numpy as np
import torch
import cv2
from skimage.morphology import binary_opening, disk, binary_erosion
import matplotlib.pyplot as plt
import tqdm
import numpy
# from hausdorff import hausdorff_distance
from skimage.morphology import dilation, erosion, disk
from skimage import metrics
import torch
import torch.nn.functional as F
import torch.nn as nn

def dice(a, b):
    aflat = a.view(-1)
    bflat = b.view(-1)
    intersection = (aflat * bflat).sum()

    return (2 * intersection) / (aflat.sum() + bflat.sum())

def hausdorff(a, b):
    b = b.long()
    a = a.long()
    d = dilation(a, disk(5))
    e = erosion(a, disk(5))
    a = d - e

    d = dilation(b, disk(5))
    e = erosion(b, disk(5))
    b = d - e

    # plt.imshow(b)
    # plt.savefig('yes.png')

    return metrics.hausdorff_distance(a, b)

def object_dice_score(gt, pred):
    max_gt = np.unique(gt)[-1]
    max_pred = np.unique(pred)[-1]

    gt_obj = []
    pred_obj = []

    gt_s = 0
    pred_s = 0

    for i in range(1, max_gt + 1):
        object = (gt == i)
        if torch.sum(object) != 0:
            gt_s += torch.sum(object)
            gt_obj.append(object)

    for j in range(1, max_pred + 1):
        object = (pred == j)
        pred_s += torch.sum(object)
        pred_obj.append(object)

    a = 0
    for j in pred_obj:
        max_overlap = 0
        ooi = torch.zeros(j.shape)
        for i in gt_obj:
            overlap = torch.sum(i.logical_and(j))
            if overlap >= max_overlap:
                max_overlap = overlap
                ooi = i
        a += (torch.sum(j) / pred_s) * dice(ooi, j)

    b = 0
    for i in gt_obj:
        max_overlap = 0
        ooi = torch.zeros(i.shape)
        for j in pred_obj:
            overlap = torch.sum(i.logical_and(j))
            if overlap >= max_overlap:
                max_overlap = overlap
                ooi = j
        b += (torch.sum(i) / gt_s) * dice(ooi, i)

    return (0.5 * (a + b))

def object_hausdorff_score(gt, pred):
    max_gt = np.unique(gt)[-1]
    max_pred = np.unique(pred)[-1]

    gt_obj = []
    pred_obj = []

    gt_s = 0
    pred_s = 0

    for i in range(1, max_gt + 1):
        object = (gt == i)
        if torch.sum(object) != 0:
            gt_s += torch.sum(object)
            gt_obj.append(object)

    for j in range(1, max_pred + 1):
        object = (pred == j)
        pred_s += torch.sum(object)
        pred_obj.append(object)

    a = 0
    for j in pred_obj:
        max_overlap = 0
        ooi = torch.zeros(j.shape)
        for i in gt_obj:
            overlap = torch.sum(i.logical_and(j))
            if overlap >= max_overlap:
                max_overlap = overlap
                ooi = i
        a += (torch.sum(j) / pred_s) * hausdorff(ooi, j)

    b = 0
    for i in gt_obj:
        max_overlap = 0
        ooi = torch.zeros(i.shape)
        for j in pred_obj:
            overlap = torch.sum(i.logical_and(j))
            if overlap >= max_overlap:
                max_overlap = overlap
                ooi = j
        
        if max_overlap == 0:
            miN = 10e5
            for j in pred_obj:
                miN = min(miN, hausdorff(j, i))
            b += (torch.sum(i) / gt_s) * miN
            continue

        b += (torch.sum(i) / gt_s) * hausdorff(ooi, i)

    return (0.5 * (a + b))

def object_f1_score(gt, pred):
    max_gt = np.unique(gt)[-1]
    max_pred = np.unique(pred)[-1]

    object_gt = []
    object_pred = []

    for i in range(1, max_gt + 1):
        object = (gt == i)
        if torch.sum(object) != 0:
            object_gt.append(object)

    for j in range(1, max_pred + 1):
        object = (pred == j)
        object_pred.append(object)

    tp = 0
    fp = 0

    lenGt = len(object_gt)

    for j in object_pred:
        match = 0
        for i in object_gt:
            gt_count = torch.sum(i == 1)
            overlap = torch.sum(j.logical_and(i))
            if(overlap / gt_count >= 0.5):
                match = 1
                break

        if match == 1:
            tp += 1
        else:
            fp += 1

    fn = lenGt - tp
    precision = tp / float(tp + fp) if tp > 0 else 0
    recall = tp / float(tp + fn) if tp > 0 else 0

    if (precision + recall) == 0:
        return 0

    # return 2 * precision * recall / (precision + recall), tp, fp, fn, precision, recall
    return 2 * precision * recall / (precision + recall)

def generate(model, img, x_th = 0.5, y_th = 0.5):
    img = img.unsqueeze(0)
    ps, pc = model(img)
    ps = ps.squeeze(0).squeeze(0).cpu().detach().numpy()
    pc = pc.squeeze(0).squeeze(0).cpu().detach().numpy()
    # ps = ps.squeeze(0)
    # pc = pc.squeeze(0)
    # ps = torch.argmax(ps, 0).cpu().detach().numpy()
    # pc = torch.argmax(pc, 0).cpu().detach().numpy()
    ps_t = np.where(ps > x_th, 1, 0)
    pc_t = np.where(pc > y_th, 1, 0)
    
    res1 = ps_t == 1
    res2 = pc_t == 0
    res = res1 & res2

    # nol, labels = cv2.connectedComponents(res.astype('uint8'))
    
    analysis = cv2.connectedComponentsWithStats(res.astype('uint8'), 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    output = np.zeros(res.shape, dtype="uint8")

    count = 1
    for i in range(1, totalLabels):

        area = values[i, cv2.CC_STAT_AREA]
        if area > 1300:
            componentMask = (label_ids == i).astype("uint8") * count
            output = output + componentMask
            count = count + 1
    
    return output, ps, pc, ps_t, pc_t

def eval(model, ds, device, x_th = 0.8, y_th = 0.5):
    total = 0
    total1 = 0
    total2 = 0
    aa = ds.__len__()
    for k in tqdm.tqdm(range(aa), 'Evaluating : '):
        a, b, c, d = ds.__getitem__(k)
        a = a.to(torch.float).to(device)
        output, ps, pc, ps_t, pc_t = generate(model, a, x_th, y_th)
        output = torch.from_numpy(output)
        a1 = object_f1_score(d, output)
        a2 = object_dice_score(d, output)
        # a3 = object_hausdorff_score(d, output)
        total += a1
        total1 += a2
        # total2 += a3

    total = total / aa
    total1 /= aa
    # total2 /= aa
    return total, total1

def box_plot(model, ds, device, x_th = 0.5, y_th = 0.5, fn = 'plot.png'):
    f1 = []
    dice = []
    hausdorff = []

    aa = ds.__len__()
    for k in tqdm.tqdm(range(aa), 'Creating....'):
        a, b, c, d = ds.__getitem__(k)
        a = a.to(torch.float).to(device)
        output, ps, pc, ps_t, pc_t = generate(model, a, x_th, y_th)
        output = torch.from_numpy(output)
        a1 = object_f1_score(d, output)
        a2 = object_dice_score(d, output)
        a3 = object_hausdorff_score(d, output)
        f1.append(a1)
        dice.append(a2)
        hausdorff.append(a3)
    f1 = np.array(f1)
    dice = np.array(dice)
    hausdorff = np.array(hausdorff)

    data = [f1, dice, hausdorff]
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)
    plt.show()
    plt.savefig(fn)
    return data

def dse_eval(model, unet, ds, device, device_1, x_th = 0.8, y_th = 0.5):
    total = 0
    total1 = 0
    aa = ds.__len__()
    for k in tqdm.tqdm(range(aa), 'Evaluating : '):
        a, b, c, d = ds.__getitem__(k)
        a = a.to(torch.float).to(device)
        output, ps, pc, ps_t, pc_t = generate(model, a, x_th, y_th)
        ps_t = torch.from_numpy(ps_t).unsqueeze(0).unsqueeze(0).to(device)
        dse_ip = torch.concat((a.unsqueeze(0), ps_t), dim = 1)
        dse_op = unet(dse_ip)
        dse_op = torch.argmax(dse_op, dim = 1).squeeze(0)
        output = torch.from_numpy(output)
        a = output.clone()
        output[torch.where(dse_op == 1)] = 1
        output[torch.where(dse_op == 2)] = 0
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(dse_op.cpu())
        ax[1].imshow(a.cpu().numpy())
        ax[2].imshow(output.cpu())
        plt.savefig('test.png')
        a1 = object_f1_score(d, output)
        a2 = object_dice_score(d, output)
        total += a1
        total1 += a2

    total = total / aa
    total1 /= aa
    return total, total1

def visualize(model, start, ds, x_th = 0.8, y_th = 0.5, fn = 'test.png'):
    fig, ax = plt.subplots(5, 8, figsize = (25, 25))
    ax[0, 0].set_title('Image')
    ax[0, 1].set_title('Segmentation (GT)')
    ax[0, 2].set_title('Segmentation (Predicted)')
    ax[0, 3].set_title('Contours (GT)')
    ax[0, 4].set_title('Contours (Predicted)')
    ax[0, 5].set_title('Ground Truth')
    ax[0, 6].set_title('Predicted')
    ax[0, 7].set_title('Metric')
    for k in range(5):
        a, b, c, e = ds.__getitem__(start + k)
        output, x, y, x1, y1 = generate(model, a, x_th, y_th)
        ax[k, 0].imshow(a.squeeze(0).permute(1, 2, 0))
        ax[k, 1].imshow(b)
        ax[k, 2].imshow(x1)
        ax[k, 3].imshow(c)
        ax[k, 4].imshow(y1)
        ax[k, 5].imshow(e)
        ax[k, 6].imshow(output)
        f1, tp, fp, fN, precision, recall = object_f1_score(gt=(e), pred=torch.from_numpy(output))
        dice = object_dice_score(gt=(e), pred=torch.from_numpy(output))
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        ax[k, 7].plot(x, y)
        ax[k, 7].text(1, 4, "F1 Score : " + str(f1) + "\nTP : {}, FP : {}, FN : {}, P : {}, R : {}".format(tp, fp, fN, precision, recall) + "\nDice Score : " + str(dice))

    plt.savefig(fn)

def dice_loss(y_true, y_pred, epsilon=1e-6): 
    axes = tuple(range(2, len(y_pred.shape))) 
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    
    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=1, gamma=3, device = torch.device('cuda:1')):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        at = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1)).unsqueeze(0)
        pt = torch.exp(-BCE_loss)
        # print(at.shape, pt.shape, BCE_loss.shape)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()