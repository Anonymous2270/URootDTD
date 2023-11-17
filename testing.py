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
from data.__init__ import name_dict
import random
import os
from torchvision import transforms
from CNNs.baselines import dtd_z_plus as sa_base
from CNNs import dtd_opt as sa_opt
from utils.visualization import visualize_cam

print(torch.__version__, torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)
# plt.figure(dpi=1000)
# plt.figure(figsize=(30, 6))

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
    class_indices = predictions.data.topk(10, dim=1)[1][0].tolist()
    max_str_len = 0
    # class_names = []
    # for cls_idx in class_indices:
    #     class_names.append(name_dict[cls_idx])
    #     if len(name_dict[cls_idx]) > max_str_len:
    #         max_str_len = len(name_dict[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, name_dict[cls_idx])
        output_string += ' ' * (max_str_len - len(name_dict[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)
    return class_indices


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


# # dog
# # catdog, 243: 'bull mastiff'
# # dogbird, 161: 'basset',  88: macaw
# # dogcat2, 207: 'golden retriever', 285, 'Egyptian cat'
# # el1, 386: 'African elephant', 340: 'zebra'
# # el2, 386: 'African elephant', 340: 'zebra'
# # el3, 386: 'African elephant', 340: 'zebra'
# # el4, 340: 'zebra', 386: 'African elephant'
# # el4, 340: 'zebra', 386: 'African elephant'


if __name__ == "__main__":
    file = open('/data1/zhangxin/Datasets/ImageNet/a.txt', 'w')
    data_path = '/data1/zhangxin/Datasets/ImageNet/train'
    file_names = []
    for root, dirs, files in os.walk(data_path):
        for sub_dir in dirs:
            if sub_dir not in [
                # 'n01443537', 'n01537544', 'n01582220', 'n01614925', 'n02113712',
                # 'n02690373',
                # 'n03126707', 'n02981792', 'n01514859'
                'A_test'
            ]:

                continue
            imgs = file_scanf2(path=data_path + '/' + sub_dir,
                               contains=[
                                   'n',
                                   # 'n01443537_9651', 'n01443537_3032', 'n01443537_1219', 'n01443537_6620',
                                   # 'n01443537_11018', 'n01443537_11018', 'n01443537_16422', 'n01443537_21526',
                                   # 'n01537544_2375', 'n01582220_392', 'n01614925_2400', 'n02113712_10565',
                                   # 'n02690373_4225'
                               ],
                               endswith='.png',
                               is_random=True, sub_ratio=1)
            file_names.extend(imgs)

    model1 = resnet18(pretrained=True).cuda()
    model1.train(False)

    act_net_base = sa_base.ActivationStoringNet(sa_base.model_flattening(model1)).cuda()
    DTD = sa_base.DTD().cuda()

    act_net_opt = sa_opt.ActivationStoringNet(sa_opt.model_flattening(model1)).cuda()
    OptDTD = sa_opt.DTDOpt().cuda()

    for f in file_names:
        file.write(str(f) + '\n')
        image = Image.open(f)
        if image.mode != 'RGB':
            continue
        re_img = resize_func(image)
        input_tensor = normal_func(re_img)
        input_tensor = torch.autograd.Variable(input_tensor.unsqueeze(0).cuda())

        module_stack1, output1 = act_net_base(input_tensor)
        # output = model(dog_cat_image.unsqueeze(0).cuda())
        # _ = model_t(dog_cat_image.unsqueeze(0).cuda())
        module_stack2, output2 = act_net_opt(input_tensor)

        class_indices = print_top_classes(output1)
        pre_idx = class_indices[0]
        # pre_dir = dir_dict[pre_idx]
        # true_dir = f.split('/')[-2]
        # true_idx = idx_dict[true_dir]
        true_idx = 243
        if true_idx == pre_idx:
            title = 'Y'
        else:
            title = 'N'

        # cat - the predicted class
        saliency_map1 = DTD(module_stack1, output1, 1000, 'resnet', index=None)
        saliency_map2 = OptDTD(module_stack2, output2, 1000, 'resnet', index=None)

        # second_cam = ignite_relprop(model_t, input=dog_cat_image.unsqueeze(0).cuda(),
        #                             method="transformer_attribution",
        #                             index=class_indices[0]).detach()
        # argmax_cam = ignite_relprop(model, input=dog_cat_image.unsqueeze(0).cuda(),
        #                             method="transformer_attribution",
        #                             index=class_indices[0]).detach()
        vis1 = visualize_cam(saliency_map1, mean_cropping=True)
        vis2 = visualize_cam(saliency_map2, mean_cropping=True)
        # sig8, root8 = OptDTD.get_sn_maps()
        # sig1, sig3, sig5, sig7, sig9, sig11 = OptDTD.get_signals()

        fig, axs = plt.subplots(3, 1)
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        axs[0].imshow(re_img)
        axs[0].axis('off')
        # axs[0].set_title(title + '-' + name_dict[true_idx])
        # axs[1].imshow(vis2)
        # axs[1].axis('off')
        # axs[1].set_title('Ours')
        # axs[2].imshow(sig1)
        # axs[2].axis('off')
        # axs[2].set_title('1')
        # axs[3].imshow(sig3)
        # axs[3].axis('off')
        # axs[3].set_title('3')
        # axs[4].imshow(sig5)
        # axs[4].axis('off')
        # axs[4].set_title('5')
        # axs[5].imshow(sig7)
        # axs[5].axis('off')
        # axs[5].set_title('7')
        # axs[6].imshow(sig9)
        # axs[6].axis('off')
        # axs[6].set_title('9')
        # axs[7].imshow(sig11)
        # axs[7].axis('off')
        # axs[7].set_title('11')
        axs[1].imshow(vis1)
        axs[1].axis('off')
        # axs[1].set_title('DTD')
        axs[2].imshow(vis2)
        axs[2].axis('off')
        # axs[2].set_title('Ours')
        # axs[2].imshow(sig8)
        # axs[2].axis('off')
        # # axs[2].set_title('Signal')
        # axs[3].imshow(root8)
        # axs[3].axis('off')
        # axs[3].set_title('Root')

        # axs[2].set_title(CLS2IDX[class_indices[1]])

        save_name = f.split('/')[-1].replace('.JPEG', '_')
        plt.savefig('/data1/zhangxin/Datasets/ImageNet/test_cnn'
                    '/' + save_name + ".jpg", dpi=500)
        plt.clf()
        # plt.close('all')
    file.close()
