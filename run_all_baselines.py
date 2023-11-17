# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/9/29 20:26
 @desc:
"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
import glob
import platform
from CNNs.resnet import resnet18
from data.__init__ import name_dict, idx_dict
import random
import os
from torchvision import transforms
from CNNs.baselines import lrp_0 as lrp_0
from CNNs.baselines import lrp_g as dtd_g
from CNNs.baselines import lrp_v as dtd_v
from CNNs.baselines import dtd_ww as dtd_ww
from CNNs.baselines import dtd_z_b as dtd_zb
from CNNs.baselines import dtd_z_plus as dtd_zp
from CNNs.baselines import dtd_monvaton as dtd_m
from CNNs import dtd_opt as sa_opt
from utils.visualization import visualize_cam

print(torch.__version__, torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
resize_func = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])
normal_func = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_idxs = predictions.data.topk(10, dim=1)[1][0].tolist()
    max_str_len = 0
    print('Top 5 classes:')
    for cls_idx in class_idxs:
        output_string = '\t{} : {}'.format(cls_idx, name_dict[cls_idx])
        output_string += ' ' * (max_str_len - len(name_dict[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    return class_idxs


def file_scanf2(path, contains, endswith, is_random=False, sub_ratio=1.0):
    files = glob.glob(path + '/*')
    if is_random:
        random.shuffle(files)
    input_files = []
    for f in files[:int(len(files) * sub_ratio)]:
        if platform.system().lower() == 'windows':
            f.replace('\\', '/')
        if not any([c in f.split('/')[-1] for c in contains]):
            continue
        if not f.endswith(endswith):
            continue
        input_files.append(f)
    return input_files


if __name__ == "__main__":
    file = open('/data1/zhangxin/Datasets/ImageNet/a.txt', 'w')
    data_path = '/data1/zhangxin/Datasets/ImageNet/train'
    file_names = []
    for root, dirs, files in os.walk(data_path):
        for sub_dir in dirs:
            # if sub_dir != 'n01443537':
            #     continue
            imgs = file_scanf2(path=data_path + '/' + sub_dir,

                               contains=[
                                   'n',
                                   # 'n01443537_9651', 'n01443537_3032', 'n01443537_1219', 'n01443537_6620',
                                   # 'n01443537_11018', 'n01443537_11018', 'n01443537_16422', 'n01443537_21526',
                                   # 'n01537544_2375', 'n01582220_392', 'n01614925_2400', 'n02113712_10565',
                                   # 'n02690373_4225',
                               ],
                               endswith='.JPEG',
                               is_random=True, sub_ratio=0.01)
            file_names.extend(imgs)

    model = resnet18(pretrained=True).cuda()
    model.train(False)

    DTDZp = dtd_zp.DTD().cuda()
    DTDZb = dtd_zb.DTD(lowest=-1., highest=1).cuda()
    DTDm = dtd_m.DTD().cuda()
    DTD0 = lrp_0.DTD().cuda()
    DTDg = dtd_g.DTD().cuda()
    DTDv = dtd_v.DTD().cuda()
    DTDww = dtd_ww.DTD().cuda()
    act_net_base1 = lrp_0.ActivationStoringNet(lrp_0.model_flattening(model)).cuda()
    act_net_base2 = dtd_g.ActivationStoringNet(dtd_g.model_flattening(model)).cuda()
    act_net_base3 = dtd_v.ActivationStoringNet(dtd_v.model_flattening(model)).cuda()
    act_net_base4 = dtd_zp.ActivationStoringNet(dtd_zp.model_flattening(model)).cuda()
    act_net_base5 = dtd_zb.ActivationStoringNet(dtd_zb.model_flattening(model)).cuda()
    act_net_base6 = dtd_m.ActivationStoringNet(dtd_m.model_flattening(model)).cuda()
    act_net_base7 = dtd_ww.ActivationStoringNet(dtd_ww.model_flattening(model)).cuda()

    act_net_opt = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model)).cuda()
    OptDTD = sa_opt.DTDOpt().cuda()

    for f in file_names:
        file.write(str(f) + '\n')
        image = Image.open(f)
        if image.mode != 'RGB':
            continue
        re_img = resize_func(image)
        input_tensor = normal_func(re_img)
        input_tensor = torch.autograd.Variable(input_tensor.unsqueeze(0).cuda())

        module_stack1, output1 = act_net_base1(input_tensor)
        module_stack2, output2 = act_net_base2(input_tensor)
        module_stack3, output3 = act_net_base3(input_tensor)
        module_stack4, output4 = act_net_base4(input_tensor)
        module_stack5, output5 = act_net_base5(input_tensor)
        module_stack6, output6 = act_net_base6(input_tensor)
        module_stack7, output7 = act_net_base7(input_tensor)
        module_stack_our, output_our = act_net_opt(input_tensor)

        class_indices = print_top_classes(output1)
        pre_idx = class_indices[0]
        # pre_dir = dir_dict[pre_idx]
        true_dir = f.split('/')[-2]
        true_idx = idx_dict[true_dir]
        if true_idx == pre_idx:
            title = 'Y'
        else:
            title = 'N'

        # cat - the predicted class
        saliency_0 = DTD0(module_stack1, output1, 1000, 'resnet')
        saliency_g = DTDg(module_stack2, output2, 1000, 'resnet')
        saliency_v = DTDv(module_stack3, output3, 1000, 'resnet')
        saliency_zp = DTDZp(module_stack4, output4, 1000, 'resnet')
        saliency_zb = DTDZb(module_stack5, output5, 1000, 'resnet')
        saliency_m = DTDm(module_stack6, output6, 1000, 'resnet')
        saliency_ww = DTDww(module_stack7, output7, 1000, 'resnet')
        saliency_our = OptDTD(module_stack_our, output_our, 1000, 'resnet')

        vis_0 = visualize_cam(saliency_0, mean_cropping=False)
        vis_g = visualize_cam(saliency_g, mean_cropping=False)
        vis_v = visualize_cam(saliency_v, mean_cropping=False)
        vis_zp = visualize_cam(saliency_zp, mean_cropping=True)
        vis_zb = visualize_cam(saliency_zb, mean_cropping=False)
        vis_m = visualize_cam(saliency_m, mean_cropping=False)
        vis_ww = visualize_cam(saliency_ww, mean_cropping=False)
        vis_our = visualize_cam(saliency_our, mean_cropping=True)
        # vis3, vis4 = OptDTD.get_feature_map()

        fig, axs = plt.subplots(1, 9)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 调整子图间距
        axs[0].imshow(re_img)
        axs[0].axis('off')
        axs[0].set_title(title + '-' + name_dict[true_idx])
        # axs[1].imshow(vis1, cmap="turbo", clim=(0, 1))
        axs[1].imshow(vis_0)
        axs[1].axis('off')
        axs[1].set_title('0')
        axs[2].imshow(vis_g)
        axs[2].axis('off')
        axs[2].set_title('g')
        axs[3].imshow(vis_v)
        axs[3].axis('off')
        axs[3].set_title('v')
        axs[4].imshow(vis_ww)
        axs[4].axis('off')
        axs[4].set_title('ww')
        axs[5].imshow(vis_zp)
        axs[5].axis('off')
        axs[5].set_title('z+')
        axs[6].imshow(vis_zb)
        axs[6].axis('off')
        axs[6].set_title('zb')
        axs[7].imshow(vis_m)
        axs[7].axis('off')
        axs[7].set_title('m')
        axs[8].imshow(vis_our)
        axs[8].axis('off')
        axs[8].set_title('Ours')

        save_name = f.split('/')[-1].replace('.JPEG', '_')
        plt.savefig('/data1/zhangxin/Datasets/ImageNet/test_cnn'
                    '/' + save_name + ".jpg", dpi=300)
        plt.clf()
        # plt.close('all')
    file.close()
