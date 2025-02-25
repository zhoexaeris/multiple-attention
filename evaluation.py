import os
import torch
import json
import numpy as np
from datasets.dataset import DeepfakeDataset
from models.MAT import MAT
from sklearn.metrics import roc_auc_score as AUC
import pickle
import re

# Load the trained model and its configuration
def load_model(name):
    with open(f'runs/{name}/config.pkl', 'rb') as f:
        config = pickle.load(f)

    net = MAT(**config.net_config)
    return config, net

# Find the best checkpoint based on accuracy
def find_best_ckpt(name, last=False):
    if last:
        return len(os.listdir(f'checkpoints/{name}')) - 1

    with open(f'runs/{name}/train.log') as f:
        lines = f.readlines()[1::2]  # Extract lines containing accuracy data

    accs = [float(re.search(r'acc\:(.*)\,', a).groups()[0]) for a in lines]
    best = accs.index(max(accs))
    return best

# Compute classification accuracy based on a fixed threshold
def acc_eval(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    thres = 0.5
    acc = np.mean((preds >= thres) == labels)
    return thres, acc

# Evaluate the trained model on the test dataset
def test_eval(net, setting, testset):
    test_dataset = DeepfakeDataset(phase='test', **setting)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=setting['batch_size'],
                                              shuffle=False, pin_memory=True, num_workers=8)

    for i, (X, y) in enumerate(test_loader):
        testset[i].append([])

        if -1 in y:  # Handle empty/corrupt samples
            testset[i].append(0.5)
            continue

        X = X.to('cuda', non_blocking=True)

        with torch.no_grad():
            for x in torch.split(X, 20):
                logits = net(x)
                pred = torch.nn.functional.softmax(logits, dim=1)[:, 1]  # Extract probability of 'fake'
                testset[i][-1] += pred.cpu().numpy().tolist()

        testset[i].append(np.mean(testset[i][-1]))

# Compute performance metrics (Accuracy, AUC) based on test predictions
def test_metric(testset):
    frame_labels, frame_preds, video_labels, video_preds = [], [], [], []

    for i in testset:
        frame_preds += i[2]
        frame_labels += [i[1]] * len(i[2])
        video_preds.append(i[3])
        video_labels.append(i[1])

    video_thres, video_acc = acc_eval(video_labels, video_preds)
    frame_thres, frame_acc = acc_eval(frame_labels, frame_preds)
    video_auc = AUC(video_labels, video_preds)
    frame_auc = AUC(frame_labels, frame_preds)

    return {
        'video_acc': video_acc,
        'video_threshold': video_thres,
        'video_auc': video_auc,
        'frame_acc': frame_acc,
        'frame_threshold': frame_thres,
        'frame_auc': frame_auc
    }

# Evaluate the trained model on a specific dataset
def all_eval(name, dataset, ckpt=None):
    config, net = load_model(name)
    setting = config.val_dataset
    setting['datalabel'] = dataset
    setting['min_frames'] = None  # No need for frame limit
    setting['frame_interval'] = None  # No need for interval
    setting['imgs_per_video'] = None  # Since we are using pre-extracted frames

    if ckpt is None:
        ckpt = find_best_ckpt(name)
    if ckpt < 0:
        ckpt = len(os.listdir(f'checkpoints/{name}')) + ckpt

    state_dict = torch.load(f'checkpoints/{name}/ckpt_{ckpt}.pth')['state_dict']
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    net.cuda()

    os.makedirs(f'evaluations/{name}', exist_ok=True)

    result = {}

    testset = DeepfakeDataset(phase='test', datalabel=dataset, resize=config.resize, normalize=config.normalize)

    test_eval(net, setting, testset)

    with open(f'evaluations/{name}/{dataset}-test-{ckpt}.json', 'w') as f:
        json.dump(testset, f)

    result[dataset] = test_metric(testset)

    with open(f'evaluations/{name}/metrics-{dataset}-{ckpt}.json', 'w') as f:
        json.dump(result, f)

# Evaluate the mean correlation between attention maps
def eval_meancorr(name, dataset, ckpt=None):
    config, net = load_model(name)
    setting = config.val_dataset
    setting['datalabel'] = dataset
    setting['frame_interval'] = None
    setting['imgs_per_video'] = 60

    if ckpt is None:
        ckpt = find_best_ckpt(name)
    if ckpt < 0:
        ckpt = len(os.listdir(f'checkpoints/{name}')) + ckpt

    state_dict = torch.load(f'checkpoints/{name}/ckpt_{ckpt}.pth')['state_dict']
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    net.cuda()

    testset = []
    test_dataset = DeepfakeDataset(phase='test', datalabel=dataset, resize=config.resize, normalize=config.normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, pin_memory=True, num_workers=8)

    count = 0
    mc_count = 0

    for i, (X, y) in enumerate(test_loader):
        x = X.to('cuda', non_blocking=True)
        with torch.no_grad():
            count += x.shape[0]
            layers = net.net(x)
            raw_attentions = layers[config.attention_layer]
            attention_maps = net.attentions(raw_attentions).flatten(-2)
            srs = torch.norm(attention_maps, dim=2)

            for a in range(0, config.num_attentions - 1):
                for b in range(a + 1, config.num_attentions):
                    mc_count += torch.sum(torch.sum(attention_maps[:, a, :] * attention_maps[:, b, :], dim=-1) / (srs[:, a] * srs[:, b]))

    return mc_count / ((config.num_attentions - 1) * config.num_attentions / 2) / count
