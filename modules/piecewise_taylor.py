# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import einops

zero_points = {'gelu': 0.0, 'relu': 1e-9, 'softmax': -5, 'zero': 0.0}
# zero_slope = {'gelu': 0.5, 'softmax': 0.001}
unfold_op = torch.nn.Unfold(kernel_size=(3, 3), padding=(1, 1))


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)  # set the min bound, means get larger than 1e-9, the "stabilizer"
    den = den + den.eq(0).type(den.type()) * 1e-9  # if den==0 then +1*1e-9
    return a / den * b.ne(0).type(b.type())  # do / if b=!0 or *0


def dtd_linear_piecewise_sample(x, w, z, under_R, R, root_zero, func=None):
    # inn = torch.zeros_like(Z)  # [b token e_out]
    # Z1 = F.linear(x1, w1)  # y=R=[b token e_out]
    # Z2 = F.linear(x2, w2)  #
    S = safe_divide(R, under_R)  # R/Zj
    delta, media_roots = root_routing_sample(x, root_zero)

    grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
    relevance = delta * grad_x

    for root in media_roots:
        z = func(root, w)
        grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
        relevance = relevance + delta * grad
    return relevance


def dtd_conv_piecewise_sample(x, w, stride, padding, z, under_R, R, root_zero, func=None):
    S = safe_divide(R, under_R)  # R/Zj
    delta, media_roots = root_routing_sample(x, root_zero)
    grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
    relevance = delta * grad_x

    for root in media_roots:
        z = F.conv2d(root, w, stride=stride, padding=padding)
        grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
        relevance = relevance + delta * grad
    return relevance


def dtd_op_piecewise_sample(x, z, under_R, R, root_zero, func=None):
    S = safe_divide(R, under_R)  # R/Zj
    delta, media_roots = root_routing_sample(x, root_zero)

    grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
    relevance = delta * grad_x

    for root in media_roots:
        z = func(root)
        grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
        relevance = relevance + delta * grad
    return relevance


def dtd_gather_piecewise_sample(x1, x2, root1, root2, z1, z2, under_R, R, func=None):
    S = safe_divide(R, under_R)  # R/Zj

    delta, media_roots = root_routing_sample(x1, root1)
    grad_x = torch.autograd.grad(outputs=z1, inputs=x1, grad_outputs=S)[0]
    relevance1 = delta * grad_x
    for root in media_roots:
        z1 = func([root, x2])
        grad = torch.autograd.grad(outputs=z1, inputs=root, grad_outputs=S)[0]
        relevance1 = relevance1 + delta * grad

    delta, media_roots = root_routing_sample(x2, root2)
    grad_x = torch.autograd.grad(outputs=z2, inputs=x2, grad_outputs=S)[0]
    relevance2 = delta * grad_x
    for root in media_roots:
        z2 = func([x1, root])
        grad = torch.autograd.grad(outputs=z2, inputs=root, grad_outputs=S)[0]
        relevance2 = relevance2 + delta * grad

    return relevance1, relevance2


def dtd_act_piecewise_sample(x, z, under_R, R, root_zero, func=None, step=50):
    S = safe_divide(R, under_R)  # R/Zj
    delta, media_roots = root_routing_sample(x, root_zero, step=step)

    grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]
    relevance = delta * grad_x

    for root in media_roots:
        if func is F.softmax:
            z = func(root, dim=-1)
        else:
            z = func(root)
        grad = torch.autograd.grad(outputs=z, inputs=root, grad_outputs=S)[0]
        relevance = relevance + delta * grad

    # if funcs[0] is F.softmax:
    #     z1 = funcs[0](root1, dim=-1)
    #     z2 = funcs[0](root2, dim=-1)
    #     z3 = funcs[0](root3, dim=-1)
    #     z4 = funcs[0](root4, dim=-1)
    #
    # else:
    #     z1 = funcs[0](root1)
    #     z2 = funcs[0](root2)
    #     z3 = funcs[0](root3)
    #     z4 = funcs[0](root4)
    #
    # grad1 = torch.autograd.grad(outputs=z1, inputs=root1, grad_outputs=S)[0]
    # grad2 = torch.autograd.grad(outputs=z2, inputs=root2, grad_outputs=S)[0]
    # grad3 = torch.autograd.grad(outputs=z3, inputs=root3, grad_outputs=S)[0]
    # grad4 = torch.autograd.grad(outputs=z4, inputs=root4, grad_outputs=S)[0]
    # grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=S)[0]

    # if grad3.equal(grad1):
    #     pass
    # else:
    #     print('_act not equal')

    # relevance = (root2 - root1) * (grad1 + grad2 + grad3 + grad4 + grad_x)
    return relevance


def root_routing_sample(x, root_t_s, step=10):
    # b, t, e = x.size()  # [b t e]
    if not torch.is_tensor(root_t_s):
        root_zero = torch.zeros_like(x, requires_grad=True).to(x.device)
        root_zero = root_zero + zero_points[root_t_s] + 1e-9
    else:
        root_zero = root_t_s

    media_roots = []
    delta = (x - root_zero) / step
    for i in range(1, step):
        media_roots.append(root_zero + delta * i)
        # delta = (x - root_zero) / 5
    # root1 = root_zero + delta
    # root2 = root1 + delta
    # root3 = root2 + delta
    # root4 = root3 + delta
    return delta, media_roots


def root_search_edge(x: torch.Tensor, dx=None, zero_point='zero'):
    thr0 = 0.001
    thr_ave = 0.001
    thr_edge = 0
    stride = 0.1

    _, _, h, _ = x.size()  # [b e h w]
    x_neighbor = unfold_op(x)  # [b e*3*3 h*w]
    x_neighbor = einops.rearrange(x_neighbor, 'b (e k) l -> b e l k', k=9)  # [b c=e 14*14 k=9]
    x_nei_copy = x_neighbor.clone().cpu().detach().numpy()
    del x_neighbor

    g_neighbor = unfold_op(dx)  # [b e*3*3 h*w]
    g_neighbor = einops.rearrange(g_neighbor, 'b (e k) l -> b e l k', k=9)  # [b c=e 14*14 k=9]
    g_nei_copy = g_neighbor.clone().cpu().detach().numpy()
    del g_neighbor

    b, c, l, k = np.shape(x_nei_copy)  # [b c l 9]
    roots = np.zeros((b, c, l), dtype=x_nei_copy.dtype)
    # APASmilkov et al. (2017). Smoothgrad: removing noise by adding noise.
    # 考虑到这种急剧的噪音波动，在任何给定的像素点的直接梯度就不如局度平均梯度值来得有意义。
    grads_roots = np.zeros((b, c, l), dtype=g_nei_copy.dtype)

    for bs in range(b):
        for ch in range(c):
            for tok in range(l):
                root = zero_points[zero_point]
                grad_root = g_nei_copy[bs, ch, tok, 4]
                pixel_x = x_nei_copy[bs, ch, tok, 4]

                if (pixel_x - root) <= thr0:  # x is very closed to the root
                    pass

                else:
                    pixel_a = x_nei_copy[bs, ch, tok, 0]
                    pixel_b = x_nei_copy[bs, ch, tok, 1]
                    pixel_d = x_nei_copy[bs, ch, tok, 3]

                    grad_b = g_nei_copy[bs, ch, tok, 1]
                    grad_d = g_nei_copy[bs, ch, tok, 3]

                    if pixel_a - np.max([pixel_b, pixel_d]) >= thr_edge:
                        root = np.min([pixel_b, pixel_d])
                        grad_root = np.min([grad_b, grad_d])
                        # root = x_nei_copy[bs, ch, tok, 8]
                        # grad_root = g_nei_copy[bs, ch, tok, 8]
                        # print('situ edge gao')
                    elif np.min([pixel_b, pixel_d]) - pixel_a >= thr_edge:
                        root = np.max([pixel_b, pixel_d])
                        grad_root = np.max([grad_b, grad_d])
                        # root = pixel_a
                        # grad_root = g_nei_copy[bs, ch, tok, 0]
                        # print('situ edge di')

                    else:
                        pixel_c = x_nei_copy[bs, ch, tok, 2]
                        if (np.abs(pixel_a - pixel_b) <= thr_ave and np.abs(pixel_b - pixel_c) <= thr_ave
                                and np.abs(pixel_a - pixel_d) <= thr_ave):
                            root = np.sum(x_nei_copy[bs, ch, tok, :4], axis=-1, keepdims=False) / 4
                            grad_root = np.sum(g_nei_copy[bs, ch, tok, :4], axis=-1, keepdims=False) / 4
                            # if grad_root / (root + 1e-07) >= 0.3:
                            #     delta = float(stride * (grad_root / np.abs(grad_root)))
                            #     root = root * (1 - delta)
                            #     # root_grad = root_grad  # * (1 - stride)
                            # else:
                            #     pass
                            #     # root2[bs, ch, tok] = 0
                            #     # brg_grad[bs, ch, tok] = 0
                        else:
                            # kk = x_nei_copy[bs, ch, tok, :]
                            # kk = einops.rearrange(kk, '(h w) -> w h', h=3)
                            # # root[bs, ch, tok] = cv2.dct(kk)[0][0]
                            # root[bs, ch, tok] = np.mean(adct_img(kk))
                            root = pixel_b + pixel_d - pixel_a
                            grad_root = (grad_b + grad_d - g_nei_copy[bs, ch, tok, 0])
                            # # root[bs, ch, tok] = 0
                            # # min_idx = np.argmin(g_neighbor[bs, ch, tok, :4])
                            # if grad_root / (root + 1e-07) >= 0.3:
                            #     delta = float(stride * (grad_root / np.abs(grad_root)))
                            #     root = root * (1 - delta)
                            #     # root_grad = root_grad  # * (1 - stride)
                            # else:
                            #     root = 0
                roots[bs, ch, tok] = root
                grads_roots[bs, ch, tok] = grad_root
            pass
        pass
    pass

    del x_nei_copy, g_nei_copy
    roots = einops.rearrange(roots, 'b c (h w) -> b (h w) c', h=h)
    roots = torch.tensor(roots, dtype=x.dtype, device=x.device)
    grads_roots = einops.rearrange(grads_roots, 'b c (h w) -> b (h w) c', h=h)
    grads_roots = torch.tensor(grads_roots, dtype=dx.dtype, device=dx.device)
    return roots, grads_roots


def get_img_roots(x, dx, activation='GELU'):
    # obtain root by average the 5*5 surround
    dims = len((x.size()))
    if dims == 4:  # image for conv: [b c h w]
        root, grads_roots = root_search_edge(x=x)
        return root, grads_roots

    if dims == 3:  # time-series for linear: [b t e]
        # cls_token = torch.select(x, dim=1, index=0).unsqueeze(0)  # root=0
        x_img = torch.index_select(x, dim=1, index=torch.arange(1, 197).to(x.device))
        x_img = einops.rearrange(x_img, 'b (h w) e -> b e h w', h=14, w=14)  # reshape to image [b e 14 14]

        g_cls = torch.select(dx, dim=1, index=0).unsqueeze(0)  # root=0
        g_img = torch.index_select(dx, dim=1, index=torch.arange(1, 197).to(dx.device))
        g_img = einops.rearrange(g_img, 'b (h w) e -> b e h w', h=14, w=14)  # reshape to image [b e 14 14]

        root, grads_roots = root_search_edge(x=x_img, dx=g_img)

        root = torch.cat([torch.zeros_like(g_cls), root], dim=1)
        grads_roots = torch.cat([g_cls, grads_roots], dim=1)

        del x_img, g_img, g_cls
        return root, grads_roots

    if dims == 2:  # output layers of FC: [b e]
        # print(x.size(), 'output')
        # x = x.unsqueeze(axis=1)
        # w = torch.ones(1, 1, 3, requires_grad=False).cuda()  # one 5*5 kernel which has 3 channels
        # neighbor_sum = F.conv1d(x, weight=w, bias=None, stride=1, padding='same')
        # re = x - (neighbor_sum / 3)
        # return re.squeeze()
        return torch.zeros_like(x), torch.ones_like(x)
    else:
        print(dims)
