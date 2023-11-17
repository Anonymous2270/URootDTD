import argparse
import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# def ignite_relprop(model, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
#     model.eval()
#     output = model(input)
#     kwargs = {"alpha": 1}
#     if index is None:
#         index = np.argmax(output.cpu().data.numpy(), axis=-1)
#
#     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#     one_hot[0, index] = 1
#     one_hot_vector = one_hot
#     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#     one_hot = torch.sum(one_hot.cuda() * output)
#
#     model.zero_grad()
#     one_hot.backward(retain_graph=True)
#
#     return model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
#                          start_layer=start_layer, **kwargs)


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_TURBO)
    heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    return heatmap


def visualize_full_cam(original_image, cam):
    cam = torch.sum(cam, dim=1)[0]
    original_image = original_image[0]

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # normalize
    m = cam.gt(cam.mean())
    cam = cam * m

    cam = cam.data.cpu().numpy()

    # cam = np.maximum(0, cam) * 255 * 8000
    # cam = np.minimum(255, cam)
    # cam = gaussian_filter(cam, sigma=3)  # 高斯平滑
    # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    img = original_image.permute(1, 2, 0).data.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    vis = show_cam_on_image(img, cam)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def visualize_cam(cam, mean_cropping=True):
    cam = torch.sum(cam, dim=1)[0]

    if not mean_cropping:
        cam = torch.clamp(cam, min=0)  # serves LRP and DTD rules, otherwise it would not see anything
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        # mask = cam.gt(cam.mean())
        # cam = cam * mask
        cam = cam.data.cpu().numpy()
        cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_TURBO)
        cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)
    else:
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        # mean cropping serves z+ rule, otherwise it would not see anything
        mask = cam.gt(cam.mean())
        cam = cam * mask
        cam = cam.data.cpu().numpy()
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
        cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)

    return cam


def visualize_featuremap(root, name=None):
    # # [1, 3, 224, 224]
    # print(' '+name, root.shape, torch.sum(root))
    # cam = torch.sum(root, dim=1)[0]
    # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # normalize
    # mask = cam.gt(cam.mean())
    # cam = cam * mask
    #
    # cam = cam.data.cpu().numpy()
    # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # normalize
    #
    # cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    # cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)
    # [1, 3, 224, 224]
    cam = torch.sum(root, dim=1)[0]

    cam = cam.data.cpu().numpy()
    cam = gaussian_filter(cam, sigma=1)  # 高斯平滑
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)
    return cam