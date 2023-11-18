# -*- coding: utf-8 -*-
import einops
import torch
from tqdm import tqdm
import numpy as np
from CNNs.resnet import resnet18
# from CNNs import dtd_opt as sa_opt
# from CNNs.baselines import lrp_0 as sa_base
# from CNNs.baselines import lrp_g as sa_base
# from CNNs.baselines import lrp_v as sa_base
# from CNNs.baselines import dtd_ww as sa_base
# from CNNs.baselines import dtd_z_b as sa_base
# from CNNs.baselines import dtd_z_plus as sa_base
from CNNs.baselines import dtd_monvaton as sa_base
from data.imagenet import ImagenetDataset

torch.manual_seed(2022)
torch.cuda.manual_seed(2022)

y_target = 1


def vis_process(vis, device):  # [b, 3, 224, 224]
    # visualization method 1
    vivi = []
    for v in vis:
        # v = torch.sum(v, dim=0, keepdim=False)
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)  # normalize
        m = v.gt(v.mean())
        v = v * m
        v = v.cpu().data.numpy()
        vivi.append(v)
    vis = np.array(vivi)
    vis = torch.from_numpy(vis).to(device)

    # # visualization method 2 放大了微小值
    # vis = torch.nn.functional.sigmoid(vis)

    # # visualization method 3
    # # Note: LRPs needs this to amplify those weak pixel correlations, which is suspicious of cheating
    # # vis = torch.clamp(vis, min=0) * 255 * 7000
    # # vis = torch.clamp(vis, max=255)
    # vis_color = []
    # for v in vis:
    #     v = torch.sum(v, dim=0)
    #     v = gaussian_filter(v.cpu().data.numpy(), sigma=1)
    #     v = (v - v.min()) / (v.max() - v.min() + 1e-9)  # normalize
    #     v = cv2.applyColorMap(np.uint8(255 * v), cv2.COLORMAP_TURBO)  # TURBO 线性， JET 减弱为微小值
    #     v = np.float32(v) / 255
    #     vis_color.append(v)
    # vis = np.array(vis_color)
    # vis = torch.from_numpy(vis).to(device)
    # vis = einops.rearrange(vis, 'b h w c -> b c h w')
    # vis = torch.clamp(vis, min=0.1)

    return vis


def calculate_loss(logits, label):
    pred = torch.softmax(logits, dim=-1)
    y = torch.nn.functional.one_hot(label, num_classes=1000).float()
    y = torch.softmax(y, dim=-1)
    loss = torch.nn.functional.cross_entropy(pred, y)
    return loss


def under_constrain_clip(perturb, l2, l9):
    norm9 = torch.abs(torch.max(perturb))
    if norm9 > l9:
        perturb = torch.clamp(perturb, -l9, l9)
    norm2 = torch.abs(perturb)
    if norm2 > l2:
        perturb = perturb * (l2 / norm2)
    return perturb


def attack_batch(model, loader, norm=2):
    net = sa_base.ActivationStoringNet(sa_base.model_flattening(model)).cuda()
    DTD = sa_base.DTD().cuda()
    # net = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    # DTD = sa_opt.DTDOpt().cuda()

    l0 = 500  # 扰动的像素点数量
    # l2 = 224 * 224 * 0.1  # 原始图像和扰动之后的图像之间，所有像素点距离绝对值的总和
    epsilon = 0.05  # l-inf, 最大扰动幅度, epsilon=[2,10]/255 in paper
    alpha = 1  # step size
    iteration = 30  # 迭代次数

    attacks = 0
    hit = 0
    hit_step = [0]
    samples = 0
    corrects_before = 0
    corrects_post = 0

    iterator = tqdm(loader)
    for batch_idx, (x, label) in enumerate(iterator):
        origin_img = x.clone()
        x = torch.autograd.Variable(x, requires_grad=True)  # [b, 3, 224, 224]
        label = label.to(x.device)
        attack = False

        # model inference
        module_stack, out = net(x)
        probs = torch.softmax(out, dim=1)
        _, _label = torch.max(probs, 1)

        samples += 1
        if _label == label:
            attack = True
            attacks += 1
            corrects_before += 1

        step = 0
        while (step < iteration) and (_label == label):
            vis = DTD(module_stack, out, 1000, 'resnet')  # [b, 3, 224, 224]
            vis = vis_process(vis, x.device)  # [b, 3, 224, 224]

            # target_logit = torch.index_select(out, dim=-1, index=label)
            # gard = torch.autograd.grad(target_logit, x, torch.ones_like(target_logit))[0]
            loss = calculate_loss(out, label)
            loss.backward()
            xg = x.grad
            if norm == 2:
                lp_g = xg / (torch.sqrt(torch.sum(xg * xg, dim=(1, 2, 3), keepdim=True)) + 10e-8)
            else:
                lp_g = xg.sign()

            v_fla = einops.rearrange(vis, 'b c h w -> b c (h w)')
            x_fla = einops.rearrange(x, 'b c h w -> b c (h w)')
            gard_fla = einops.rearrange(lp_g, 'b c h w -> b c (h w)')

            _, idx = torch.topk(v_fla, k=l0, dim=-1)  # [b c l0]
            val = torch.gather(x_fla, dim=-1, index=idx)
            grad = torch.gather(gard_fla, dim=-1, index=idx)

            # x_fla = x_fla.data.scatter_(dim=-1, index=idx, src=val * (1 - torch.sign(xg) * epsilon))
            x_fla = x_fla.data.scatter_(dim=-1, index=idx, src=val + grad * alpha)
            x = einops.rearrange(x_fla, 'b c (h w) -> b c h w', h=224, w=224)

            # project current example back onto Lp ball
            if norm == 2:
                d = x - origin_img
                ad = torch.abs(d)
                mask = ad <= epsilon  # true if satisfy l2
                scale = d.clone()
                scale[mask] = epsilon
                _d = d * epsilon / scale
                x = origin_img + d
                # d = x - origin_img
                # mask = epsilon >= d.view(d.shape[0], -1).norm(2, dim=1)
                # scale = d.view(d.shape[0], -1).norm(2, dim=1)
                # scale[mask] = epsilon
                # d *= epsilon / scale.view(-1, 1, 1, 1)
                # x = origin_img + d
            elif norm in ["inf", np.inf]:
                x = torch.max(torch.min(x, origin_img + epsilon), origin_img - epsilon)
            x = x.clamp(-1.0, 1.0)

            # model inference
            x = x.clone().detach().requires_grad_(True)
            net.zero_grad()
            module_stack, out = net(x)
            probs = torch.softmax(out, dim=1)
            _, _label = torch.max(probs, 1)

            step += 1

        if _label != label and attack:
            hit += 1
            hit_step.append(step)
        if _label == label and attack:
            corrects_post += 1

        iterator.set_description('Hit: %.2f, attacks: %.2f, aveStep: %.1f, accB: %.2f, accP: %.2f' %
                                 (hit / attacks, attacks, np.array(hit_step).mean(), corrects_before/samples,
                                  corrects_post / samples))


if __name__ == "__main__":
    imagenet_ds = ImagenetDataset(img_transfer=True)

    model = resnet18(pretrained=True).cuda()
    model.eval()

    loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=1,
        shuffle=False)

    attack_batch(model, loader)
