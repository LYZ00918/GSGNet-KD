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
from toolbox.loss.kd_losses.sp import SP
from toolbox.loss.kd_losses.vid import VID
from toolbox.loss.kd_losses.pkt import PKTCosSim
from toolbox.loss.kd_losses.at import AT
from toolbox.loss.kd_losses.ofd import OFD
from toolbox.loss.kd_losses.fsp import FSP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# teacher model
from toolbox.models.LYZ.lyz_2 import MyConv_resnet_T
from toolbox.models.LYZ.lyz_stu import Mobilenet_S
# from toolbox.models.LYZ.lyz_test_S3 import Mobilenet_S
# from toolbox.models.LYZ.lyz_test_1 import Mobilenet_S

'''
just test loss
'''


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
# weight = ClassWeight('median_freq_balancing')
criterion = nn.CrossEntropyLoss().cuda()
criterion1 = nn.KLDivLoss().cuda()
criterion2 = SP().cuda()
criterion3 = PKTCosSim().cuda()
criterion4 = AT(2).cuda()
# 16/24/32/160/320
criterion5_1 = VID(64,5).cuda()
criterion5_2 = VID(128,5).cuda()
criterion5_3 = VID(256,5).cuda()
criterion5_4 = VID(512,5).cuda()
criterion5_5 = VID(512,5).cuda()

criterion6_1 = OFD(16,64).cuda()
criterion6_2 = OFD(24,128).cuda()
criterion6_3 = OFD(32,256).cuda()
criterion6_4 = OFD(160,512).cuda()
criterion6_5 = OFD(320,512).cuda()

criterion7 = FSP().cuda()
criterion_without = MscCrossEntropyLoss().cuda()
# criterion1 = nn.CrossEntropyLoss(weight=weight.get_weight(test_dataloader, num_classes=5)).cuda()

criterion_focal1 = FocalLossbyothers().cuda()
criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督
net_s = Mobilenet_S().cuda()
net_T = MyConv_resnet_T().cuda()
net_T.load_state_dict(torch.load('./weight/MyConv_resnet_T_1-Potsdam-loss.pth'))
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

        # [12, 1, 256, 256]
        # print(edge.shape)
        # label_student = F.interpolate(label.unsqueeze(1).float(), size=size).squeeze(1).long().cuda()
        # teacher, student = net(image, ndsm)
        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image, ndsm)
        # out_s1,out_s2,out_s3,out_s4,out_s5,out_s6 = net(image)
        # out = net(image)
        # out_t1,out_t2,out_t3,out_t4,out_t5,out_t6 = net_T(image)
        out1 = net_T(image,ndsm)
        # with teacher transformer to label
        teacher_out = out1[0].data.cpu().numpy()
        teacher_out = np.argmax(teacher_out,axis=1)
        teacher_out = torch.from_numpy(teacher_out)
        teacher_out = Variable(teacher_out.long().cuda())
        loss = criterion_without(out[0],teacher_out)
        # loss calculate
        loss1 = criterion_without(out[0:6], label)  # 没有边界监督

        # tout1 = F.softmax(out1[0]/t,dim=1)
        # tout2 = F.softmax(out1[1]/t,dim=1)
        # tout3 = F.softmax(out1[2]/t,dim=1)
        # tout4 = F.softmax(out1[3]/t,dim=1)
        # tout5 = F.softmax(out1[4]/t,dim=1)
        # tout6 = F.softmax(out1[5]/t,dim=1)

        # sout1 = F.softmax(out[0] / t, dim=1)
        # sout2 = F.softmax(out[1] / t, dim=1)
        # sout3 = F.softmax(out[2] / t, dim=1)
        # sout4 = F.softmax(out[3] / t, dim=1)
        # sout5 = F.softmax(out[4] / t, dim=1)
        # sout6 = F.softmax(out[5] / t, dim=1)
        # loss2 = criterion1(sout1, tout1) * t * t  # 没有边界监督
        # loss3 = criterion1(sout2, tout2) * t * t  # 没有边界监督
        # loss4 = criterion1(sout3, tout3) * t * t  # 没有边界监督
        # loss5 = criterion1(sout4, tout4) * t * t  # 没有边界监督
        # loss6 = criterion1(sout5, tout5) * t * t  # 没有边界监督
        # loss7 = criterion1(sout6, tout6) * t * t  # 没有边界监督

        # loss8 = criterion2(out1[0],out[0])
        # loss9 = criterion2(out1[1],out[1])
        # loss10 = criterion2(out1[2],out[2])
        # loss11 = criterion2(out1[3],out[3])
        # with calculate semanitically loss
        # loss12 = criterion2(out1[4],out[4])
        # loss13 = criterion2(out1[5],out[5])
        # loss_4 = (loss12 + loss13) / 2
        #
        # loss14 = criterion3(out1[0],out[0])
        # loss15 = criterion3(out1[1],out[1])
        # loss16 = criterion3(out1[2],out[2])
        # loss17 = criterion3(out1[3],out[3])
        # loss18 = criterion3(out1[4],out[4])
        # loss19 = criterion3(out1[5],out[5])
        # ATLoss
        loss21 = criterion4(out1[1],out[1])
        loss22 = criterion4(out1[2],out[2])
        loss23 = criterion4(out1[3],out[3])
        loss24 = criterion4(out1[4],out[4])
        # loss25 = criterion4(out1[5],out[5])
        loss_20_1 = (loss21+loss24+loss23+loss22)/4

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
        #  ofd loss
        # loss31 = criterion6_1(out[6],out1[6])
        # loss32 = criterion6_2(out[7],out1[7])
        # loss33 = criterion6_3(out[8],out1[8])
        # loss34 = criterion6_4(out[9],out1[9])
        # loss35 = criterion6_5(out[10],out1[10])
        # loss_30 = (loss35+loss34+loss33+loss32+loss31)/5

        # loss30 = criterion7()
        # fsploss
        loss31 = criterion7(a1,a2,out1[6],out1[7])
        loss32 = criterion7(a2,a3,out1[7],out1[8])
        loss33 = criterion7(a3,a4,out1[8],out1[9])
        loss34 = criterion7(a4,a5,out1[9],out1[10])
        loss_30 = loss34 + loss33 + loss32 + loss31
        # loss_all1 = (loss6 + loss5 + loss7 + loss4 + loss3  )/5
        # loss_all3 = (loss14 + loss15 + loss17 + loss16 + loss18 + loss19 )/6
        # loss_all2 = (loss14 + loss15 + loss16 + loss17 + loss18 + loss19 )/6
        # loss3 = (loss25 + loss24 + loss23 + loss22 + loss21)/5
        # loss = criterion_without(out, label)
        # loss2 = criterion_without(out[1], label)
        # loss = loss_all1 * (1 - alpha) + loss1 + loss_all2
        # loss = loss_all1 * (1 - alpha) + loss1 + loss + loss_4 * 0.5
        # 边界监督
        # loss1 = criterion_without(out[0], label)
        # loss2 = criterion_bce(nn.Sigmoid()(out[1]), edge)
        # loss = (loss2 + loss1) / 2
        loss = loss1 + loss_20_1 + loss + loss_30  + loss_20
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
            # imageVal = F.interpolate(imageVal, (320, 320), mode="bilinear", align_corners=True)
            # ndsmVal = F.interpolate(ndsmVal, (320, 320), mode="bilinear", align_corners=True)
            # labelVal = F.interpolate(labelVal.unsqueeze(1).float(), (320, 320),
            #                          mode="bilinear", align_corners=True).squeeze(1).long()
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # teacherVal, studentVal = net(imageVal, ndsmVal)
            # outVal = net(imageVal)
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
        # torch.save(net.state_dict(), './weight/Mobilenet_S_kd_7-{}-loss.pth'.format(DATASET))
        # torch.save(net.state_dict(), './toolbox/models/LYZ/weight/Mobilenet_S_KD111-{}-loss.pth'.format(DATASET))
        torch.save(net.state_dict(), './toolbox/models/LYZ/weight/Mobilenet_S_KD222-{}-loss.pth'.format(DATASET))


    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

