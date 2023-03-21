from torch.utils.data import DataLoader
import torch
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from testClassifier_CV import testing, getPlots
import time
from matplotlib import pyplot as plt
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gpu_id = 4
map_loc = "cuda:"+str(gpu_id)

# T = 36
'------configuration:-------------------------------------------'
dataset = 'NUCLA'
Alpha = 0.1 # bi loss
lam1 = 2 # cls loss
lam2 = 1 # mse loss

N = 80 * 2
Epoch = 100
# num_class = 10
dataType = '2D'
sampling = 'Multi' #sampling strategy
fistaLam = 0.1
RHdyan = True
withMask = False
maskType = 'score'
constrastive = True

if sampling == 'Single':
    num_workers = 8
    bz = 60
else:
    num_workers = 4
    bz = 12

T = 36 # input clip length
mode = 'dy+bi+cl'
setup = 'setup1'
fusion = False
num_class = 10
 # v1,v2 train, v3 test;
lr = 1e-3 # classifier
lr_2 = 1e-4  # sparse codeing
gumbel_thresh = 0.505

# print parameters
print('gumbel thresh: ',gumbel_thresh)
print('Classifier learning rate: ',lr)
print('Transformer learning rate: ',lr_2)
print('Sparse coding learning rate: ',lr_2)
print('Alpha, lam1, lam2 ', Alpha, lam1, lam2)

'change to your own model path'
modelRoot = './models/crossView_NUCLA/'

saveModel = modelRoot + sampling + '/' + mode + '/T36_contrastive_sparse_fixed/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)
print('mode:', mode, 'sampling: ', sampling, ' model path:', saveModel,  'gpu:', gpu_id)

'============================================= Main Body of script================================================='

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()


path_list = './data/CV/' + setup + '/'
# root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
trainSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='train', cam='2,1', T=T, maskType=maskType,
                                setup=setup)
# #

trainloader = DataLoader(trainSet, batch_size=bz, shuffle=True, num_workers=num_workers)

testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='test', cam='2,1', T=T, maskType= maskType, setup=setup)
testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers)


net = contrastiveNet(dim_embed=128, Npole=N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id, dim=2, dataType='2D', fistaLam=fistaLam, fineTune=True).cuda(gpu_id)

'load pre-trained contrastive model'
#pre_train = modelRoot + sampling + '/' + mode + '/T36_contrastive_fineTune_all/' + '40.pth'        
pre_train = './pretrained/' + dataset +'/' + setup + '/' +sampling + '/pretrainedRHdyan_CL.pth' 
state_dict = torch.load(pre_train, map_location=map_loc)

net = load_fineTune_model(state_dict, net)

'frozen all layers except last classification layer'
for p in net.backbone.sparseCoding.parameters():
    p.requires_grad = False

# for p in net.backbone.transformer_encoder.parameters():
#     p.requires_grad = True

for p in net.backbone.Classifier.parameters():
    p.requires_grad = True

for p in net.backbone.Classifier.cls[-1].parameters():
    p.requires_grad = True

# optimizer = torch.optim.SGD(
#         [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': lr_2},
#         {'params': filter(lambda x: x.requires_grad, net.backbone.transformer_encoder.parameters()), 'lr': lr_2}  ,
#          {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': lr}], weight_decay=1e-3,
#         momentum=0.9)

optimizer = torch.optim.SGD(
        [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': lr_2},
         {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': lr}], weight_decay=1e-3,
        momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()

L1loss = torch.nn.SmoothL1Loss()
LOSS = []
ACC = []
LOSS_CLS = []
LOSS_MSE = []
LOSS_BI = []
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    count = 0
    pred_cnt = 0
    start_time = time.time()
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()

        skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        # skeletons = sample['input_skeletons']['unNormSkeleton'].float().cuda(gpu_id)
        # skeletons = sample['input_skeletons']['affineSkeletons'].float().cuda(gpu_id)
        images = sample['input_images'].float().cuda(gpu_id)
        ROIs = sample['input_rois'].float().cuda(gpu_id)
        visibility = sample['input_skeletons']['visibility'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)


        if sampling == 'Single':
            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)  #bz, T, 25, 2
            input_mask = visibility.reshape(visibility.shape[0], t, -1)
            nClip = 1
            input_images = images
            input_rois = ROIs

        else:
            t = skeletons.shape[2]
            input_skeletons = skeletons.reshape(skeletons.shape[0],skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
            input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
            input_rois = ROIs.reshape(ROIs.shape[0]* ROIs.shape[1], t, 3, 224, 224)
            nClip = skeletons.shape[1]

        actPred, lastFeat, binaryCode, output_skeletons = net(input_skeletons, bi_thresh=gumbel_thresh, nclips=nClip)

        if sampling == 'Single':
            actPred = actPred
            pred = torch.argmax(actPred, 1)

        else:
            act_pred = actPred.reshape(skeletons.shape[0], skeletons.shape[1], num_class)
            act_pred = torch.mean(act_pred, 1)
            pred = torch.argmax(act_pred, 1)

        bi_gt = torch.zeros_like(binaryCode).cuda(gpu_id)
        actPred = actPred.reshape(skeletons.shape[0], nClip, num_class)
        actPred = torch.mean(actPred, 1)
        target_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1],t,-1)
        loss = lam1 * Criterion(actPred, gt_label) + lam2 * mseLoss(output_skeletons, target_skeletons.squeeze(-1)) \
               + Alpha * L1loss(binaryCode, bi_gt)

        lossMSE.append(mseLoss(output_skeletons, target_skeletons.squeeze(-1)).data.item())

        lossBi.append(L1loss(binaryCode, bi_gt).data.item())

        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())

        lossCls.append(Criterion(actPred, gt_label).data.item())

        ## Train acc
        correct = torch.eq(gt_label, pred).int()
        count += gt_label.shape[0]
        pred_cnt += torch.sum(correct).data.item()

    loss_val = np.mean(np.array(lossVal))
    LOSS.append(loss_val)
    LOSS_CLS.append(np.mean(np.array((lossCls))))
    LOSS_MSE.append(np.mean(np.array(lossMSE)))
    LOSS_BI.append(np.mean(np.array(lossBi)))
    train_acc = pred_cnt/count
    print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)),
          '|bi:', np.mean(np.array(lossBi)), '|acc:', train_acc)
    end_time = time.time()
    print('training time(mins):', (end_time - start_time) / 60.0) #mins

    scheduler.step()
    if epoch % 5 == 0:
        # torch.save({'state_dict': net.state_dict(),
        #            'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

        Acc = testing(testloader, net, gpu_id, sampling, mode, withMask,gumbel_thresh)
        print('testing epoch:', epoch, 'Acc:%.4f' % Acc)
        ACC.append(Acc)

torch.cuda.empty_cache()
print('done')
