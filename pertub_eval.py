# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/11/4 16:08
 @desc:
"""
import cv2
import einops
import torch
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from CNNs.resnet import resnet18
# from CNNs import dtd_opt as sa_opt
# from CNNs.baselines import dtd_monvaton as sa_base
# from CNNs.baselines import dtd_z_plus as sa_base
from CNNs.baselines import dtd_ww as sa_base
# from CNNs.baselines import dtd_z_b as sa_base
from data.imagenet import ImagenetDataset

positive_perturbation = True
experiment_dir = '/data1/zhangxin/Datasets/ImageNet/test_cnn'


def eval(model, loader):
    net = sa_base.ActivationStoringNet(sa_base.model_flattening(model)).cuda()
    DTD = sa_base.DTD().cuda()
    # net = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    # DTD = sa_opt.DTDOpt().cuda()

    num_samples = 0
    num_correct = np.zeros((len(imagenet_ds,)))
    # dissimilarity = np.zeros((len(imagenet_ds,)))
    model_index = 0

    base_size = 224 * 224
    perturbation_steps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_correct_pertub = np.zeros((10, len(imagenet_ds)))
    # dissimilarity_pertub = np.zeros((10, len(imagenet_ds)))
    # logit_diff_pertub = np.zeros((10, len(imagenet_ds)))
    # prob_diff_pertub = np.zeros((10, len(imagenet_ds)))
    perturb_index = 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        # Update the number of samples
        num_samples += len(data)

        target = target.to(data.device)

        # model inference
        module_stack, output = net(data.clone())
        vis = DTD(module_stack, output, 1000, 'resnet')  # [b, 3, 224, 224]

        # vivi = []
        # for v in vis:
        #     v = torch.sum(v, dim=0, keepdim=False)
        #     v = (v - v.min()) / (v.max() - v.min())  # normalize
        #     # v = v / torch.max(torch.abs(v))
        #     m = v.gt(v.mean())
        #     v = v * m
        #     v = v.cpu().data.numpy()
        #     vivi.append(v)
        # vis = np.array(vivi)
        # vis = torch.from_numpy(vis).to(data.device)
        # vis = torch.sum(vis, dim=1)
        # vis = torch.clamp(vis, min=0)

        # # pertub method 2 放大了微小值
        # vis = torch.nn.functional.sigmoid(vis)

        # pertub method 3
        vis_color = []
        for v in vis:
            v = torch.sum(v, dim=0)
            # v = torch.clamp(v, min=0)
            # v = gaussian_filter(v.cpu().data.numpy(), sigma=1)
            v = v.cpu().data.numpy()
            v = (v - v.min()) / (v.max() - v.min())  # normalize
            v = cv2.applyColorMap(np.uint8(255 * v), cv2.COLORMAP_TURBO)  # TURBO
            v = np.float32(v) / 255
            vis_color.append(v)
        vis = np.array(vis_color)
        vis = torch.from_numpy(vis).to(data.device)
        vis = einops.rearrange(vis, 'b h w c -> b c h w')
        vis = torch.sum(vis, dim=1)

        # pred_probabilities = torch.softmax(output, dim=1)  # [b, 1000]
        # pred_org_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)  # [b]
        # pred_org_logit = output.data.max(1, keepdim=True)[0].squeeze(1)  # [b]

        pred_class = output.data.max(1, keepdim=True)[1].squeeze(1)  # [b]
        tgt_pred = (target == pred_class).type(target.type()).data.cpu().numpy()   # [b, ]
        num_correct[model_index:model_index+len(tgt_pred)] = tgt_pred

        # probs = torch.softmax(output, dim=1)
        # target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        # second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        # temp = torch.log(target_probs / second_probs).data.cpu().numpy()
        # dissimilarity[model_index:model_index+len(temp)] = temp

        # if wrong:
        #     wid = np.argwhere(tgt_pred == 0).flatten()
        #     if len(wid) == 0:
        #         continue
        #     wid = torch.from_numpy(wid).to(vis.device)
        #     vis = vis.index_select(0, wid)
        #     data = data.index_select(0, wid)
        #     target = target.index_select(0, wid)

        # Save original shape
        org_shape = data.shape

        # vis = vis.reshape(org_shape[0], -1)
        vis = einops.rearrange(vis, 'b h w -> b (h w)')

        for i in range(len(perturbation_steps)):
            _data = data.clone()

            _, idx = torch.topk(vis, k=int(base_size * perturbation_steps[i]),
                                largest=positive_perturbation, dim=-1)
            idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
            _data = _data.reshape(org_shape[0], org_shape[1], -1)
            _data = _data.scatter_(-1, idx, 0)
            _data = _data.reshape(*org_shape)

            out = model(_data)

            # pred_probabilities = torch.softmax(out, dim=1)
            # pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)

            # diff = (pred_prob - pred_org_prob).data.cpu().numpy()
            # prob_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            # pred_logit = out.data.max(1, keepdim=True)[0].squeeze(1)
            # diff = (pred_logit - pred_org_logit).data.cpu().numpy()
            # logit_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            target_class = out.data.max(1, keepdim=True)[1].squeeze(1)
            temp = (target == target_class).type(target.type()).data.cpu().numpy()
            num_correct_pertub[i, perturb_index:perturb_index+len(temp)] = temp

            # probs_pertub = torch.softmax(out, dim=1)
            # target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
            # second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
            # temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            # dissimilarity_pertub[i, perturb_index:perturb_index+len(temp)] = temp

        model_index += len(target)
        perturb_index += len(target)

    print('num_correct\n', np.mean(num_correct), np.std(num_correct))
    # print('dissimilarity\n', np.mean(dissimilarity), np.std(dissimilarity))
    print('\n', perturbation_steps)
    print('num_correct_pertub\n', np.mean(num_correct_pertub, axis=1), np.std(num_correct_pertub, axis=1))
    # print('dissimilarity_pertub\n', np.mean(dissimilarity_pertub, axis=1), np.std(dissimilarity_pertub, axis=1))


if __name__ == "__main__":
    imagenet_ds = ImagenetDataset()

    model = resnet18(pretrained=True).cuda()
    model.train(False)

    loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=64,
        shuffle=True)

    eval(model, loader)
