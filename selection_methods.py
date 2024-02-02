import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
# from config import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
import time
import pdb

def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD
                

def get_consistency(models, unlabeled_loader, trial, cycle, args):
    models['backbone_swa'].eval()
    num_students = (args.num_epochs-args.swa_start)//args.lr_interval 
    # if args.num_students >= num_students:
    #     pass
    # else:
    #     num_students = args.num_students
    
    with torch.no_grad():
        ST_consistency = torch.zeros((args.subset_size, num_students)).cuda()

    for i in range(num_students):
        if i==0:
            pass
        else:
            checkpoint = torch.load(args.log_root + '/t' + str(trial) + 'c' + str(cycle) + 'e'
                                    + str(args.num_epochs-i*args.lr_interval) + '.ckpt')
            models['backbone'].load_state_dict(checkpoint['state_dict'])
        models['backbone'].eval()

        with torch.no_grad():
            consistency = torch.tensor([]).cuda()
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(args.gpu):
                    inputs = inputs.cuda() # [128, 3, 32, 32]
                pred, _, _ = models['backbone'](inputs)
                pred_swa, _, _ = models['backbone_swa'](inputs)
                pred_softmax = torch.softmax(pred, dim=1)
                pred_swa_softmax = torch.softmax(pred_swa, dim=1)

                # temp : [0~1] do sharpening
                if args.sharpening_temp:
                    pt = torch.pow(pred_swa_softmax, (1 / args.sharpening_temp))
                    pred_swa_softmax = pt / torch.sum(pt, dim=1, keepdim=True)
                kl_val = torch.sum(pred_swa_softmax * torch.log(pred_swa_softmax / (pred_softmax + 1e-14)), dim=1)
                consistency = torch.cat((consistency, kl_val), 0)
        ST_consistency[:, i] = consistency
    return torch.mean(ST_consistency, dim=1).cpu()


def get_consistency_ema(models, unlabeled_loader, trial, cycle, args):
    models['backbone_ema'].eval()
    num_students = (args.num_epochs-args.swa_start)//args.lr_interval 
    # if args.num_students >= num_students:
    #     pass
    # else:
    #     num_students = args.num_students
    
    with torch.no_grad():
        ST_consistency = torch.zeros((args.subset_size, num_students)).cuda()

    for i in range(num_students):
        if i==0:
            pass
        else:
            checkpoint = torch.load(args.log_root + '/t' + str(trial) + 'c' + str(cycle) + 'e'
                                    + str(args.num_epochs-i*args.lr_interval) + '.ckpt')
            models['backbone'].load_state_dict(checkpoint['state_dict'])
        models['backbone'].eval()

        with torch.no_grad():
            consistency = torch.tensor([]).cuda()
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(args.gpu):
                    inputs = inputs.cuda() # [128, 3, 32, 32]
                pred, _, _ = models['backbone'](inputs)
                pred_ema, _, _ = models['backbone_ema'](inputs)
                pred_softmax = torch.softmax(pred, dim=1)
                pred_ema_softmax = torch.softmax(pred_ema, dim=1)

                # temp : [0~1] do sharpening
                if args.sharpening_temp:
                    pt = torch.pow(pred_ema_softmax, (1 / args.sharpening_temp))
                    pred_ema_softmax = pt / torch.sum(pt, dim=1, keepdim=True)
                kl_val = torch.sum(pred_ema_softmax * torch.log(pred_ema_softmax / (pred_softmax + 1e-14)), dim=1)
                consistency = torch.cat((consistency, kl_val), 0)
        ST_consistency[:, i] = consistency
    return torch.mean(ST_consistency, dim=1).cpu()


def get_features(models, unlabeled_loader, args):
    if args.SWA_inference:
        models['backbone_swa'].eval()
    else:
        models['backbone'].eval()
    with torch.cuda.device(args.gpu):
        features = torch.tensor([]).cuda()    
    with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(args.gpu):
                    inputs = inputs.cuda()
                    if args.SWA_inference:
                        _, features_batch, _ = models['backbone_swa'](inputs)
                    else:
                        _, features_batch, _ = models['backbone'](inputs)
                features = torch.cat((features, features_batch), 0)
            feat = features
    return feat

def get_embedding(models, unlabeled_loader, args):
    if args.SWA_inference:
        models['backbone_swa'].eval()
    else:
        models['backbone'].eval()
    with torch.cuda.device(args.gpu):
        badge_embeddings = torch.zeros((0, args.num_classes*args.num_features)).cuda()
    with torch.no_grad():
            for inputs, targets, _ in unlabeled_loader:
                badge_embedding = torch.zeros(inputs.shape[0], args.num_classes*args.num_features).cuda()
                with torch.cuda.device(args.gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    if args.SWA_inference:
                        pred, features_batch, _ = models['backbone_swa'](inputs)
                    else:
                        pred, features_batch, _ = models['backbone'](inputs)
                softmax = torch.softmax(pred, dim=1)
                for j in range(inputs.shape[0]):
                    for cls in range(args.num_classes):
                        if cls == targets[j]:
                            badge_embedding[j, args.num_features*cls: args.num_features*(cls+1)] = features_batch[j] * (1 - softmax[j, cls])
                        else:
                            badge_embedding[j, args.num_features*cls: args.num_features*(cls+1)] = features_batch[j] * (-softmax[j, cls])
                badge_embeddings = torch.cat((badge_embeddings, badge_embedding), 0)
    return badge_embeddings

def get_kcg(models, labeled_data_size, unlabeled_loader, subset_size, budget, args):
    if args.SWA_inference:
        models['backbone_swa'].eval()
    else:
        models['backbone'].eval()
    with torch.cuda.device(args.gpu):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            with torch.cuda.device(args.gpu):
                inputs = inputs.cuda()
            if args.SWA_inference:
                _, features_batch, _ = models['backbone_swa'](inputs)
            else:
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(subset_size,(subset_size + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, budget)
        other_idx = [x for x in range(subset_size) if x not in batch]
    return  other_idx + batch


def init_centers(X, K):
    from sklearn.metrics import pairwise_distances
    from scipy import stats
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, subset_size, budget, batch, labeled_set, trial, cycle, args):
    # sample with large value is selected (entropy, variance, consistency)
    if method == 'Random':
        arg = np.random.randint(subset_size, size=subset_size)

    elif method == 'ST-CoNAL':
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=batch,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        start_time = time.time()
        consistency = get_consistency(model, unlabeled_loader, trial, cycle, args)
        arg = torch.argsort(consistency)
        
    elif method == 'ST-CoNAL-EMA':
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=batch,
                                      sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        consistency = get_consistency_ema(model, unlabeled_loader, trial, cycle, args)
        arg = torch.argsort(consistency)

    else:
        pdb.set_trace()
        print("choose wrong method")

    return arg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
