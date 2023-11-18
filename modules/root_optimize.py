# -*- coding: utf-8 -*-
import einops
import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_ssim


def apply_func(x, paras=None):
    up_func = paras[0]
    if up_func == 'relu':
        x = F.relu(x)
    elif up_func == 'conv2d':
        x = F.conv2d(x, weight=paras[1], stride=paras[2], padding=paras[3])
    elif up_func == 'linear':
        x = F.linear(x, weight=paras[1])
    elif up_func == 'bn':
        x = x
    elif up_func == 'drop':
        x = x
    elif up_func == 'avgpool':
        x = F.avg_pool2d(x, kernel_size=paras[1]) * paras[1][0] ** 2 + 1e-9
    elif up_func == 'maxpool':
        x = x
    elif up_func == 'divide':
        x = x
    elif up_func == 'skip':
        x = x
    else:
        # up_func == None
        x = x
    return x


def rel_sup_root_2nd_cnn(x, R, step=50, the_layer=None, r_layer=None):
    lr = 0.001
    x_new = x.clone()
    # x_new = torch.nn.Parameter(x.clone(), requires_grad=True)
    # optimizer = torch.optim.SGD([x_new], lr=lr)

    if r_layer is None:  # last FC
        for _ in range(step):
            # optimizer.zero_grad()
            y = apply_func(x_new, the_layer)
            # y = F.softmax(y, dim=-1)
            # R = F.softmax(R, dim=-1)
            loss = nn.functional.cross_entropy(y, R)
            grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
            # grad_interp = torch.clamp(grad_interp, max=1)
            delta = lr * grad_interp
            x_new = x_new - delta
            # loss.backward(retain_graph=True)
            # optimizer.step()

    elif r_layer[0] == 'linear':  # 1d
        for _ in range(step):
            # optimizer.zero_grad()
            y = apply_func(apply_func(x_new, the_layer), r_layer)
            # y = F.softmax(y, dim=-1)
            # R = F.softmax(R, dim=-1)
            loss = nn.functional.cross_entropy(y, R)
            grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
            # grad_interp = torch.clamp(grad_interp, max=1)
            delta = lr * grad_interp
            x_new = x_new - delta
            # loss.backward(retain_graph=True)
            # optimizer.step()

    else:  # 2d
        for _ in range(step):
            # optimizer.zero_grad()
            y = apply_func(apply_func(x_new, the_layer), r_layer)
            # w = y.shape[-1]
            # y = einops.rearrange(y, 'b c h w -> b c (h w)')
            # R = einops.rearrange(R, 'b c h w -> b c (h w)')
            # y = F.softmax(y, dim=-1)
            # R = F.softmax(R, dim=-1)
            # y = einops.rearrange(y, 'b c (h w) -> b c h w', w=w)
            # R = einops.rearrange(R, 'b c (h w) -> b c h w', w=w)
            loss = pytorch_ssim.ssim(y, R)
            grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
            # grad_interp = torch.clamp(grad_interp, max=1)
            delta = lr * grad_interp
            x_new = x_new - delta
            # loss.backward(retain_graph=True)
            # optimizer.step()
    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    root = x - x_new
    root = root.detach()
    return root


def rel_sup_root_1st_cnn(x, R, step=50, the_layer=None, r_layer=None):
    lr = 1
    # scale_x = x.mean()
    x_new = x.clone().detach().requires_grad_(True)

    if the_layer[0] == 'linear':  # 1d
        R_ = F.softmax(R, dim=-1)
        for _ in range(step):
            x_new = torch.relu(x_new)
            y = apply_func(x_new, the_layer)
            y = F.softmax(y, dim=-1)
            loss = nn.functional.cross_entropy(y, R_)
            grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
            grad_interp = torch.clamp(grad_interp, max=1)
            delta = lr * grad_interp
            x_new = x_new - delta

    else:  # 2d
        w = R.shape[-1]
        R_ = einops.rearrange(R, 'b c h w -> b c (h w)')
        R_ = F.softmax(R_, dim=-1)
        R_ = einops.rearrange(R_, 'b c (h w) -> b c h w', w=w)
        for _ in range(step):
            x_new = torch.relu(x_new)
            y = apply_func(x_new, the_layer)
            y = einops.rearrange(y, 'b c h w -> b c (h w)')
            y = F.softmax(y, dim=-1)
            y = einops.rearrange(y, 'b c (h w) -> b c h w', w=w)
            loss = pytorch_ssim.ssim(y, R_)
            grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
            grad_interp = torch.clamp(grad_interp, max=1)
            delta = lr * grad_interp
            x_new = x_new - delta
    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    x_new = torch.relu(x_new)
    # scale_s = x_new.mean()
    # x_new = x_new * scale_x / scale_s
    root = x - x_new
    root = root.detach()
    return root


# def rel_sup_root_1st_cnn(x, R, step=100, the_layer=None):
#     lr = 20
#     x_new = torch.nn.Parameter(x.clone(), requires_grad=True)
#     optimizer = torch.optim.SGD([x_new], lr=lr)
#
#     if the_layer[0] == 'linear':  # 1d
#         for _ in range(step):
#             optimizer.zero_grad()
#             # x_new = torch.relu(x_new)
#             y = apply_func(x_new, the_layer)
#             loss = nn.functional.cross_entropy(y, R)
#             loss.backward(retain_graph=True)
#             optimizer.step()
#
#     else:  # 2d
#         for _ in range(step):
#             optimizer.zero_grad()
#             # x_new = torch.relu(x_new)
#             y = apply_func(x_new, the_layer)
#             loss = pytorch_ssim.ssim(y, R)
#             loss.backward(retain_graph=True)
#             optimizer.step()
#     # 此时可以假设认为 f(x_new)=class response, f(root)=0
#     # x_new = torch.relu(x_new)
#     root = x - x_new
#     root = root.detach()
#     return root


def rel_sup_root_1st_1d(x, R, w=None, func=None, step=50):
    x_new = x.clone()
    alpha = 0.01
    # x_new = torch.nn.Parameter(x_new, requires_grad=True)
    # optimizer = torch.optim.SGD([x_new], lr=alpha)
    # FC layer for output
    for _ in range(step):
        if w is not None:
            y = func(x_new, w)
        else:
            if func is F.softmax:
                y = func(x_new, dim=-1)
            else:
                y = func(x_new)
        y = F.softmax(y, dim=-1)
        R = F.softmax(R, dim=-1)
        loss = nn.functional.cross_entropy(y, R)
        # loss.backward(retain_graph=True)
        # optimizer.step()
        # loss = torch.pow(y - R, 2)
        grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
        delta = alpha * grad_interp
        x_new = x_new - delta
    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    root = x - x_new
    root = root.detach()
    return root


def rel_sup_root_1st_together_1d(x1, x2, R, func=None, step=50):
    x_new = x1.clone()
    alpha = 0.01
    # x_new = torch.nn.Parameter(x_new, requires_grad=True)
    # optimizer = torch.optim.SGD([x_new], lr=alpha)
    # FC layer for output
    for _ in range(step):
        y = func([x_new, x2])

        y = F.softmax(y, dim=-1)
        R = F.softmax(R, dim=-1)
        loss = nn.functional.cross_entropy(y, R)
        # loss.backward(retain_graph=True)
        # optimizer.step()
        # loss = torch.pow(y - R, 2)
        grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
        delta = alpha * grad_interp
        x_new = x_new - delta
    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    root1 = x1 - x_new
    root1 = root1.detach()

    x_new = x2.clone()
    for _ in range(step):
        y = func([x1, x_new])
        y = F.softmax(y, dim=-1)
        R = F.softmax(R, dim=-1)
        loss = nn.functional.cross_entropy(y, R)
        # loss.backward(retain_graph=True)
        # optimizer.step()
        # loss = torch.pow(y - R, 2)
        grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
        delta = alpha * grad_interp
        x_new = x_new - delta
    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    root2 = x2 - x_new
    root2 = root2.detach()

    return root1, root2


# def de_grad_root_sample2d(x, R, w=None, func=None, step=500, ups=None):
#     x_new = x.clone()
#     alpha = 0.001
#     # FC layer for output
#     for _ in range(step):
#         if w is not None:
#             y = apply_up_func(func(x_new, w), ups)
#         else:
#             y = apply_up_func(func(x_new), ups)
#         # y = einops.rearrange(y, 'b c h w -> b c (h w)')
#         # _R = einops.rearrange(R, 'b c h w -> b c (h w)')
#         # y = F.softmax(y, dim=-1)
#         # _R = F.softmax(_R, dim=-1)
#         # loss_interp = torch.pow(y - _R, 2)
#         loss_interp = pytorch_ssim.ssim(y, R)
#         grad_interp = torch.autograd.grad(outputs=loss_interp, inputs=x_new, grad_outputs=torch.ones_like(loss_interp))[0]
#         delta = alpha * grad_interp
#         x_new = x_new - delta
#
#     # 此时可以假设认为 f(x_new)=class response, f(root)=0
#     root = x - x_new
#     root = root.detach()
#     return root
#
#
# def de_grad_root_conv(x, R, w, stride, padding, step=500, ups=None):
#     x_new = x.clone()
#     alpha = 0.001
#     # FC layer for output
#     for _ in range(step):
#         y = apply_up_func(F.conv2d(x_new, w, stride=stride, padding=padding), ups)
#         # y = einops.rearrange(y, 'b c h w -> b c (h w)')
#         # _R = einops.rearrange(R, 'b c h w -> b c (h w)')
#         # y = F.softmax(y, dim=-1)
#         # _R = F.softmax(_R, dim=-1)
#         # loss_interp = torch.pow(y - _R, 2)
#         loss_interp = pytorch_ssim.ssim(y, R)
#         grad_interp = torch.autograd.grad(outputs=loss_interp, inputs=x_new, grad_outputs=torch.ones_like(loss_interp))[0]
#         delta = alpha * grad_interp
#         x_new = x_new - delta
#
#     # 此时可以假设认为 f(x_new)=class response, f(root)=0
#     root = x - x_new
#     root = root.detach()
#     return root
#
#
# def de_grad_root_nn(x, R, w=None, func=F.linear, step=100):
#     x_new = x.clone().cpu().data.numpy()
#     x_new = torch.from_numpy(x_new).float().to(x.device)
#     w_inplace = w.clone().cpu().data.numpy()
#     R_inplace = R.clone().cpu().data.numpy()
#     R_inplace = torch.from_numpy(R_inplace).float().to(R.device)
#     dgr = DeGradRoot(func, x_new)
#     dgr.train()
#     optimizer = torch.optim.Adam(dgr.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#     for _ in range(step):
#         w = torch.from_numpy(w_inplace).float().to(x.device)
#         optimizer.zero_grad()
#         y = dgr(w)
#         y = F.softmax(y, dim=-1)
#         loss_interp = torch.sum(torch.pow(y - R_inplace, 2))
#         loss_interp.backward()
#         optimizer.step()
#
#     x_new = dgr.x.detach()
#
#     # 此时可以假设认为 f(x_new)==zero
#     root = x - x_new
#     root = root.detach()
#     del optimizer, x_new, dgr, y, loss_interp, w_inplace, R_inplace
#     return root
#
#
# class DeGradRoot(torch.nn.Module):
#     def __init__(self, func=F.linear, x=None):
#         super(DeGradRoot, self).__init__()
#         self.func = func
#         self.x = torch.nn.Parameter(x, requires_grad=True)
#
#     def forward(self, w):
#         y = self.func(self.x, w)
#         return y