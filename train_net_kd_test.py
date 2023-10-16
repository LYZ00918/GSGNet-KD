import json
import torch
import os
import argparse
from torch import nn
import torch.nn.functional as F
import numpy as np
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss
from toolbox.loss.kd_losses.vid import VID
from toolbox.loss.kd_losses.at import AT
from toolbox.loss.kd_losses.fsp import FSP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# teacher model
from toolbox.models.GSGNet_T import GSGNet_T
from toolbox.models.GSGNet_S import GSGNet_S

DATASET = "Potsdam"
# DATASET = "Vaihingen"
batch_size = 10
import argparse
parser = argparse.ArgumentParser(description="config")

parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="configs/{}(256x256).json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()
with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
criterion4 = AT(2).cuda()
# 16/24/32/160/320
criterion5_1 = VID(64,5).cuda()
criterion5_2 = VID(128,5).cuda()
criterion5_3 = VID(256,5).cuda()
criterion5_4 = VID(512,5).cuda()
criterion5_5 = VID(512,5).cuda()

criterion7 = FSP().cuda()
criterion_without = MscCrossEntropyLoss().cuda()

net_s = GSGNet_S().cuda()
net_T = GSGNet_T().cuda()
net_T.load_state_dict(torch.load('./weight/GSGNet_T-Potsdam-loss.pth'))
net_T.load_state_dict(torch.load('./weight/GSGNet_T-Vaihingen-loss.pth'))
optimizer = optim.Adam(net_s.parameters(), lr=1e-4, weight_decay=5e-4)

def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
best = [0.0]
size = (56, 56)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 200
# schduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # setting the learning rate desend starage
for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    # for group in optimizer.param_groups:
    # 	group['lr'] *= 0.99
    # 	print(group['lr'])
    train_loss = 0
    net = net_s.train()
    prec_time = datetime.now()
    alpha = 0.90
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]
        # edge = Variable(sample['edge'].float().cuda())  # 边界监督 [12, 256, 256]

        # image = F.interpolate(image, (320, 320), mode="bilinear", align_corners=True)
        # ndsm = F.interpolate(ndsm, (320, 320), mode="bilinear", align_corners=True)
        # label = F.interpolate(label.unsqueeze(1).float(), (320, 320), mode="bilinear", align_corners=True).squeeze(1).long()
        # edge = F.interpolate(edge.unsqueeze(1), (224, 224), mode="bilinear", align_corners=True)

        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image, ndsm)
        with torch.no_grad():
            out1 = net_T(image,ndsm)
        # with teacher transformer to label
        teacher_out = out1[0].data.cpu().numpy()
        teacher_out = np.argmax(teacher_out,axis=1)
        teacher_out = torch.from_numpy(teacher_out)
        teacher_out = Variable(teacher_out.long().cuda())
        
        loss = criterion_without(out[0],teacher_out)
        loss1 = criterion_without(out[0:6], label)  
        # ATLoss
        loss21 = criterion4(out1[1],out[1])
        loss22 = criterion4(out1[2],out[2])
        loss23 = criterion4(out1[3],out[3])
        loss24 = criterion4(out1[4],out[4])
        loss25 = criterion4(out1[5],out[5])
        loss_AR = (loss21+loss24+loss23+loss22+loss25)/5

        s_1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1)).cuda()
        s_2 = nn.Sequential(nn.Conv2d(24, 128, kernel_size=1)).cuda()
        s_3 = nn.Sequential(nn.Conv2d(32, 256, kernel_size=1)).cuda()
        s_4 = nn.Sequential(nn.Conv2d(160, 512, kernel_size=1)).cuda()
        s_5 = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1)).cuda()
        a1 = s_1(out[6])
        a2 = s_2(out[7])
        a3 = s_3(out[8])
        a4 = s_4(out[9])
        a5 = s_5(out[10])
        # # vid loss
        loss26 = criterion5_1(a1,out1[6])
        loss27 = criterion5_2(a2,out1[7])
        loss28 = criterion5_3(a3,out1[8])
        loss29 = criterion5_4(a4,out1[9])
        loss30 = criterion5_5(a5,out1[10])
        loss_20 = (loss30+loss29+loss26+loss28+loss27)/5
       
        # fsploss
        loss31 = criterion7(a1,a2,out1[6],out1[7])
        loss32 = criterion7(a2,a3,out1[7],out1[8])
        loss33 = criterion7(a3,a4,out1[8],out1[9])
        loss34 = criterion7(a4,a5,out1[9],out1[10])
        loss_30 = loss34 + loss33 + loss32 + loss31
        loss_SRT = loss_20 + loss_30
        loss = loss1 + loss + loss_AR + loss_SRT
        # 边界监督

        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0

        optimizer.zero_grad()

        loss.backward()  # backpropagation to get gradient
        # qichuangaaaxuexi
        optimizer.step()  # update the weight

        train_loss = loss.item() + train_loss

    net = net_s.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            outVal = net(imageVal, ndsmVal)
            loss = criterion_without(outVal, labelVal)
            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            # outVal = outVal[1].max(dim=1)[1].data.cpu().numpy()
            labelVal = labelVal.data.cpu().numpy()
            accval = accuary(outVal, labelVal)
            # print('accVal:', accval)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)

    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))

    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        torch.save(net.state_dict(), './weight/GSGNet_KD-{}-loss.pth'.format(DATASET))
    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

