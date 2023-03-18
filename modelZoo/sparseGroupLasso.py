import scipy.io
import torch
import os
import time
import sys

sys.path.append('../')
sys.path.append('../data')
from utils import *
from dataset.crossView_UCLA import *
import numpy as np
import torch.nn as nn
import numpy.linalg as la
from modelZoo.sparseCoding import creatRealDictionary
from modelZoo.gumbel_module import *
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class GroupLasso(nn.Module):
    def __init__(self, lam, groups,  group_regs, gpu_id):
        super(GroupLasso, self).__init__()
        # self.Dictionary = Dictionary
        # self.input = input
        self.lam = lam
        self.groups = groups
        self.group_regs = group_regs
        self.max_iter = 100
        self.gpu_id = gpu_id

    def _l1_l2_prox(self,x_newminus, x_newplus ):
        return self._group_l2_prox(self._l1_prox(x_newminus, x_newplus))

    def _l1_prox(self, x_newminus, x_newplus):

        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)
        return x_new

    def _l2_prox(self, coeff, reg):
        """The proximal operator for reg*||coeff||_2 (not squared).
        """
        norm_coeff = torch.norm(coeff)
        if norm_coeff == 0:
            return 0 * coeff
        return max(0, 1 - reg / norm_coeff) * coeff

    def _group_l2_prox(self, coeff):
        """The proximal map for the specified groups of coefficients.
        """
        coeffs = coeff.clone()

        for group, reg in zip(self.groups, self.group_regs):
            coeffs[:,group,:] = self._l2_prox(coeff[:,group,:], reg)
        coeff = coeffs
        return coeff

    def compute_next_momentum(self, current_momentum):
        return 0.5 + 0.5 * np.sqrt(1 + 4 * current_momentum ** 2)

    def _update_step(self, x, momentum_x, momentum, A,const_xminus,const_xplus):

        # norm_x = self._group_l2_prox()

        Ay = torch.matmul(A, momentum_x)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus


        new_x = self._l1_l2_prox(x_newminus, x_newplus)  #l1_l2 regularizer
        # new_x = self._group_l2_prox(x)
        new_momentum = self.compute_next_momentum(momentum)

        dx = new_x - x
        new_momentum_x = new_x + dx * (momentum - 1) / new_momentum   #y_old

        return new_x, new_momentum_x, new_momentum

    def forward(self, Dictionary, input):
        'get optimal coeff'

        DtD = torch.matmul(torch.t(Dictionary), Dictionary)
        DtY = torch.matmul(torch.t(Dictionary), input)

        L = torch.abs(torch.linalg.eigvals(DtD)).max()
        Linv = 1 / L
        w = 1
        lambd = (w * self.lam) * Linv.data.item()

        # x0 = np.asarray(coeff)

        optimal_x = torch.randn(input.shape[0], Dictionary.shape[-1], input.shape[-1]).cuda(self.gpu_id)  #x_old
        momentum_x = optimal_x  #y_old
        momentum = 1

        const_xminus = torch.mul(DtY, Linv) - lambd
        const_xplus = torch.mul(DtY, Linv) + lambd

        A = torch.eye(DtD.shape[1]).cuda(self.gpu_id) - torch.mul(DtD, Linv)
        iter = 0

        while iter < self.max_iter:
            # print('iter:', iter)
            new_optimal_x, new_momentum_x, new_momentum = self._update_step(
                optimal_x, momentum_x, momentum, A,const_xminus,const_xplus)
            del momentum_x

            if torch.norm(new_optimal_x - optimal_x) / torch.norm(optimal_x + 1e-16) < 1e-6:
                break
            optimal_x = new_optimal_x
            momentum_x = new_momentum_x
            momentum = new_momentum

            # del new_momentum_x, new_momentum_x, new_momentum
            del new_optimal_x
            iter = iter+ 1

        return optimal_x

class GroupLassoEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, groups, group_regs, gpu_id):
        super(GroupLassoEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

        self.getSparseCode = GroupLasso(self.lam, groups,  group_regs, self.gpu_id)
    def forward(self, x, T):

        dic = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)

        sparseGL = self.getSparseCode(dic, x)

        reconstruction = torch.matmul(dic, sparseGL.cuda(self.gpu_id))

        return sparseGL,dic, reconstruction

if __name__ == '__main__':
    # dataPath = '/data/Yuexi/Cross_view/1115/binaryCode/coeff_bi_Y_v2_action05.mat'
    #
    # dict = scipy.io.loadmat('/data/Yuexi/Cross_view/UCLA_fista01_dictionary.mat')
    # Drr = torch.tensor(dict['Drr']).squeeze(0).cuda(1)
    # Dtheta = torch.tensor(dict['Dtheta']).squeeze(0).cuda(1)
    # data = scipy.io.loadmat(dataPath)
    #
    # coeff = torch.tensor(data['coeff_action05'])
    # origY = torch.tensor(data['origY_action05'])
    #
    # dictionary = creatRealDictionary(origY.shape[2], Drr, Dtheta, gpu_id=1).cpu()
    #
    # inputSeq = torch.matmul(dictionary, coeff[0])
    '==============================================================='
    modelRoot = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/'
    saveModel = modelRoot + 'Single/groupLassoDYAN_l1000_l2001/'
    # saveModel = modelRoot +
    if not os.path.exists(saveModel):
        os.makedirs(saveModel)
    lam = 0.1
    groups = np.linspace(0, 160, 161, dtype=np.int)
    group_reg = 0.1
    group_regs = torch.ones(len(groups)) * group_reg
    N = 80 * 2
    Epoch = 70
    gpu_id = 5

    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    withMask = False
    net = GroupLassoEncoder(Drr, Dtheta, lam, groups, group_regs, gpu_id)
    net.cuda(gpu_id)
    setup = 'setup1'  # v1,v2 train, v3 test;
    path_list = '../data/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling='Single', phase='train', cam='2,1', T=36,
                               maskType='score',
                               setup=setup)
    # #

    trainloader = DataLoader(trainSet, batch_size=16, shuffle=True, num_workers=8)

    testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling='Single', phase='test', cam='2,1', T=36,
                              maskType='score', setup=setup)
    testloader = DataLoader(testSet, batch_size=8, shuffle=True, num_workers=2)


    '=========================Training========================================================================='
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=1e-4,
                                momentum=0.9)

    net.train()

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    mseLoss = torch.nn.MSELoss()
    # Loss = []
    for epoch in range(1, Epoch+1):
        print('training epoch:', epoch)
        lossVal = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()

            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

            GL, _, output_skeletons = net(input_skeletons, t)

            loss = mseLoss(output_skeletons, input_skeletons)
            loss.backward()
            # print('rr.grad:', net.rr.grad, 'theta.grad:', net.theta.grad, 'GL:',GL[0,0:10,0:10])
            optimizer.step()

            lossVal.append(loss.data.item())
        end_time = time.time()
        print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)),'time(h):', (end_time-start_time)/3600)
        print('rr.grad:', net.rr.grad, 'theta.grad:', net.theta.grad, 'GL:', torch.sum(GL))
        if epoch % 10 == 0:
            # torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
            #             'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

            with torch.no_grad():
                ERROR = torch.zeros(testSet.__len__(),1)

                for i,sample in enumerate(testloader):
                    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

                    t = skeletons.shape[1]
                    input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

                    GL, _, output_skeletons = net(input_skeletons, t)

                    error = torch.norm(output_skeletons-input_skeletons).cpu()
                    ERROR[i] = error

                print('epoch:', epoch, 'error:', torch.mean(ERROR))

        scheduler.step()
    """""
    '======================================================================================='
    model_file = saveModel + '70.pth'
    state_dict = torch.load(model_file)['state_dict']
    net.load_state_dict(state_dict)
    binaryCoding = GumbelSigmoid()
    with torch.no_grad():

        for i, sample in enumerate(testloader):
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            act_label = sample['action']
            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

            GroupSparseCode, _, output_skeletons = net(input_skeletons, t)

            binaryCode = binaryCoding(GroupSparseCode**2, force_hard=True, temperature=0.1, inference=True)

            for b in range(0, binaryCode.shape[0]):
                fig = plt.figure()
                img = plt.imshow(binaryCode[b].detach().cpu().numpy())
                title = 'action_'+ str(act_label[b].data.item())
                plt.title(title)
                fig.colorbar(img)
                plt.savefig('../logfiles/1019/figs_001_005/GL_bi_'+title + '_' + str(b) + '.png')
                plt.close(fig)
            print('check')
    """""
    print('done')

