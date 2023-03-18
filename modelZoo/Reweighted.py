import torch.nn as nn
import scipy.io
import pdb
import torch
import matplotlib.pyplot as plt
from modelZoo.sparseCoding import creatRealDictionary
import numpy as np
def fista_new(D, Y, lambd, maxIter):
    DtD = torch.matmul(torch.t(D),D) #161x161
    L = torch.norm(DtD, 2)
    # L = torch.linalg.matrix_norm(DtD, 2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y) # bz x 161 x 50
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]) # 161 x 50
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())   # ---> (w*lambd)/L
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]) - torch.mul(DtD,linv) # 161x161
    DtY = torch.mul(DtY,linv) # 1 x 161 x 50
    Softshrink = nn.Softshrink(lambd)

    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old) #161x50
        del y_old
        # temp = w * (Ay + DtY)
        # x_new = Softshrink(temp)
        x_new = Softshrink(Ay+DtY)   # --> shrink(y_k)
        # x_new = lambd * (Ay+DtY)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt) # 1 x 161 x 50
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-8:
            x_old = x_new
            print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old

def fista_group(D, Y, lambd, group1, maxIter):
    DtD = torch.matmul(torch.t(D),D) #161x161
    L = torch.norm(DtD, 2)
    # L = torch.linalg.matrix_norm(DtD, 2)
    linv = 1/L
    DtY_all = torch.matmul(torch.t(D),Y) # bz x 161 x 50
    # group2  = [0,1,2,3,4,5,6,7,8,9]
    DtY1 = torch.zeros(Y.shape[0],len(group1),1) # row
    for i in range(0, len(group)):
        dty = DtY_all[:,i,:] # 1 x 161 x 1
        dtyNorm = torch.norm(dty, p=2) #1x161
        DtY1[:,i,:] = dtyNorm

    DtY = DtY1
    # DtY2 = torch.zeros(Y.shape[0], len(group2), DtY_all.shape[-1])
    #
    # for i in range(0, int(DtD.shape[-1]/len(group)), DtY_all.shape[-1]):
    #     dty = DtY_all[:, :,i:i+2] # 1 x 161 x 2
    #     dtyNorm = torch.norm(dty, p=2) #1x161
    #     DtY2[:,:,i] = dtyNorm

    # DtY = torch.cat()

    x_old = torch.zeros(Y.shape[0], DtD.shape[1],DtY.shape[2]) # 161 x 50
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())   # ---> (w*lambd)/L
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]) - torch.mul(DtD,linv) # 161x161
    DtY = torch.mul(DtY,linv) # 1 x 161 x 50
    Softshrink = nn.Softshrink(lambd)

    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old) #161x50
        del y_old
        # temp = w * (Ay + DtY)
        # x_new = Softshrink(temp)
        x_new = Softshrink(Ay+DtY)   # --> shrink(y_k)
        # x_new = lambd * (Ay+DtY)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt) # 1 x 161 x 50
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-8:
            x_old = x_new
            print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old

def fista_reweighted(D, Y, lambd, w, maxIter):
    DtD = torch.matmul(torch.t(D), D)
    DtY = torch.matmul(torch.t(D), Y)
    # eig,v = torch.eig(DtD, eigenvectors=True)
    # L = torch.max(eig)
    # L = torch.linalg.matrix_norm(DtD, 2)
    # L = torch.linalg.matrix_norm(D,2)** 2
    L = torch.abs(torch.linalg.eigvals(DtD)).max()
    Linv = 1/L
    lambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2])
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - lambd
    const_xplus = torch.mul(DtY, Linv) + lambd

    iter = 0

    while iter < maxIter:
        iter +=1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-8:
            x_old = x_new
            # print(iter)
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new
    print(iter)

    return x_old


if __name__ == '__main__':

    dataPath = '/data/Yuexi/Cross_view/1115/binaryCode/coeff_bi_Y_v2_action05.mat'
    # dictionary = scipy.io.loadmat('/home/dan/ws/2021-CrossView/matfiles/1115_coeff-wi-bin/Dictionary_clean_20211116.mat')
    dict = scipy.io.loadmat('/data/Yuexi/Cross_view/UCLA_fista01_dictionary.mat')
    Drr = torch.tensor(dict['Drr']).squeeze(0).cuda(1)
    Dtheta = torch.tensor(dict['Dtheta']).squeeze(0).cuda(1)
    data = scipy.io.loadmat(dataPath)
    # dictionary = torch.tensor(data['dictionary'])
    coeff = torch.tensor(data['coeff_action05'])
    origY = torch.tensor(data['origY_action05'])
    dictionary = creatRealDictionary(origY.shape[2],Drr,Dtheta,gpu_id=1).cpu()

    inputSeq = torch.matmul(dictionary, coeff[0])

    # group = [0,1,2,3,4,5,6,7,8,9]
    c_sparse = fista_new(dictionary, inputSeq, 0.1, 100)

    group = np.linspace(0, 160, 161, dtype=np.int)
    c_group = fista_group(dictionary, inputSeq, 50, group,200)
    c_group = c_group[0,:,0]
    # mask = c_group
    c_group[c_group!=0] = torch.ones(c_group[c_group!=0].shape)
    mask = c_group.unsqueeze(0).repeat(dictionary.shape[0],1)

    new_dict = c_group * dictionary
    c_sparse_new = fista_new(new_dict, inputSeq, 0.1, 100)


    print('check')
    # pdb.set_trace()

    # y = origY[10].unsqueeze(0)

    # w_init = torch.ones(1,161,1)
    # # c_init = torch.zeros(61,1)
    # lambd = 0.1
    #
    # c_sparse = fista_new(dictionary, y, lambd, 100)




    # i = 0

    # while i < 3:
    #
    #     c_sparse_re = fista_reweighted(dictionary, y,lambd, w_init, 200 )
    #
    #
    #     w = 1/(torch.abs(c_sparse_re) + 1e-2)
    #     w_norm = w/torch.norm(w,2)
    #     # print('iter:',i, w[0].t())
    #     w_init = w_norm * 61
    #     # w_init = w
    #         # lam_int = lam
    #     i+=1
    #     # c_init = c_sparse_re
    #
    #     c_final = c_sparse_re
    #     del c_sparse_re
    #
    # print('check')