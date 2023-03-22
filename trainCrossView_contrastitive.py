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

gpu_id = 0
map_loc = "cuda:"+str(gpu_id)

# T = 36
'------configuration:-------------------------------------------'
dataset = 'NUCLA'

N = 80 * 2
Epoch = 200
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
    bz = 64
else:
    num_workers = 8
    bz = 8

T = 36 # input clip length
mode = 'dy+bi+cl'
setup = 'setup1'

fusion = False
num_class = 10
 # v1,v2 train, v3 test;
lr = 1e-3 # classifier
lr_1 = 1e-4 # transformer
lr_2 = 1e-4  # sparse codeing
gumbel_thresh = 0.51
'change to your own model path'
modelRoot = '/home/balaji/Cross-View/Cross-View-CL/models/crossView_NUCLA/'


saveModel = modelRoot + sampling + '/' + mode + '/T36_contrastive/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)
print('mode:',mode, 'model path:', saveModel, 'mask:', maskType)

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

trainloader = DataLoader(trainSet, batch_size=bz, shuffle=True, num_workers=num_workers, drop_last=True)

testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='test', cam='2,1', T=T, maskType= maskType, setup=setup)
testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers, drop_last=True)


net = contrastiveNet(dim_embed=128, Npole=N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id, dim=2, dataType='2D', fistaLam=fistaLam, fineTune=False).cuda(gpu_id)
net.train()



# optimizer = torch.optim.SGD(
#         [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': lr_2},
#          {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': lr}], weight_decay=1e-3,
#         momentum=0.9)

optimizer = torch.optim.SGD(
        [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': lr_2},
        {'params': filter(lambda x: x.requires_grad, net.backbone.transformer_encoder.parameters()), 'lr': lr_1},
         {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': lr}], weight_decay=1e-3,
        momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.SmoothL1Loss()
cosSIM = nn.CosineSimilarity(dim=1, eps=1e-6)

LOSS = []
ACC = []

LOSS_CLS = []
LOSS_MSE = []
LOSS_BI = []
print('experiment setup:',RHdyan, constrastive)
for epoch in range(1, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []

    start_time = time.time()
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()

        # skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        # skeletons = sample['input_skeletons']['unNormSkeleton'].float().cuda(gpu_id)
        skeletons = sample['input_skeletons']['affineSkeletons'].float().cuda(gpu_id)
        images = sample['input_images'].float().cuda(gpu_id)
        ROIs = sample['input_rois'].float().cuda(gpu_id)
        visibility = sample['input_skeletons']['visibility'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)

        if sampling == 'Single':
            t = skeletons.shape[2]
            input_skeletons = skeletons.reshape(skeletons.shape[0],skeletons.shape[1], t, -1)  #bz, 2, T, 25, 2
            input_mask = visibility.reshape(visibility.shape[0], t, -1)
            nClip = 1
            input_images = images
            input_rois = ROIs

        else:
            t = skeletons.shape[3]
            input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], skeletons.shape[2], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
            input_mask = visibility.reshape(visibility.shape[0]*visibility.shape[1], t, -1)
            input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
            input_rois = ROIs.reshape(ROIs.shape[0]*ROIs.shape[1], t, 3, 224, 224)
            nClip = skeletons.shape[1]

        # info_nce_loss = net(input_skeletons, t)
        # print('input_skeletons shape ', input_skeletons.shape)
        logits, labels = net(input_skeletons, bi_thresh=gumbel_thresh, nclips=nClip)
        info_nce_loss = Criterion(logits, labels)
        info_nce_loss.backward()
        optimizer.step()
        lossVal.append(info_nce_loss.data.item())
    scheduler.step()
    print('epoch:', epoch, 'contrastive loss:', np.mean(np.asarray(lossVal)))
    # print('rr.grad:', net.backbone.sparseCoding.rr.grad, 'cls grad:', net.backbone.Classifier.cls[-1].weight.grad[0:10,0:10])
    if epoch % 5 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

'plotting results:'
# getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC,fig_name='DY_CL.pdf')
torch.cuda.empty_cache()
print('done')