# -*- coding: utf-8 -*-
from CNNs.resnet import BasicBlock, Bottleneck
from modules.piecewise_taylor import *
from modules.root_optimize import *
from utils.visualization import visualize_featuremap


def model_flattening(module_tree):
    module_list = []
    children_list = list(module_tree.children())
    if len(children_list) == 0 or isinstance(module_tree, BasicBlock) or \
            isinstance(module_tree, Bottleneck):
        return [module_tree]
    else:
        for i in range(len(children_list)):
            module = model_flattening(children_list[i])
            module = [j for j in module]
            module_list.extend(module)
        return module_list


class ActivationStoringNet(nn.Module):
    def __init__(self, module_list):
        super(ActivationStoringNet, self).__init__()
        self.module_list = module_list

    def basic_block_forward(self, basic_block, activation):
        identity = activation

        basic_block.conv1.activation = activation
        activation = basic_block.conv1(activation)
        activation = basic_block.relu(basic_block.bn1(activation))
        basic_block.conv2.activation = activation
        activation = basic_block.conv2(activation)
        activation = basic_block.bn2(activation)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample)):
                basic_block.downsample[i].activation = identity
                identity = basic_block.downsample[i](identity)
            basic_block.identity = identity
        basic_block.activation = activation
        output = activation + identity
        output = basic_block.relu(output)

        return basic_block, output

    def bottleneck_forward(self, bottleneck, activation):
        identity = activation

        bottleneck.conv1.activation = activation
        activation = bottleneck.conv1(activation)
        activation = bottleneck.relu(bottleneck.bn1(activation))
        bottleneck.conv2.activation = activation
        activation = bottleneck.conv2(activation)
        activation = bottleneck.relu(bottleneck.bn2(activation))
        bottleneck.conv3.activation = activation
        activation = bottleneck.conv3(activation)
        activation = bottleneck.bn3(activation)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample)):
                bottleneck.downsample[i].activation = identity
                identity = bottleneck.downsample[i](identity)
            bottleneck.identity = identity
        bottleneck.activation = activation
        output = activation + identity
        output = bottleneck.relu(output)

        return bottleneck, output

    def forward(self, x):
        module_stack = []
        activation = x

        for i in range(len(self.module_list)):
            module = self.module_list[i]
            if isinstance(module, BasicBlock):
                module, activation = self.basic_block_forward(module, activation)
                module_stack.append(module)
            elif isinstance(module, Bottleneck):
                module, activation = self.bottleneck_forward(module, activation)
                module_stack.append(module)
            else:
                module.activation = activation
                module_stack.append(module)
                activation = module(activation)
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    activation = activation.view(activation.size(0), -1)

        output = activation

        return module_stack, output


class DTDOpt(nn.Module):
    def __init__(self):
        super(DTDOpt, self).__init__()
        self.signal_map_1 = None
        self.signal_map_3 = None
        self.signal_map_5 = None
        self.signal_map_7 = None
        self.signal_map_9 = None
        self.signal_map_11 = None
        self.root_map = None

    def get_sn_maps(self):
        self.signal_map_7 = visualize_featuremap(self.signal_map_7)
        self.root_map = visualize_featuremap(self.root_map)
        return self.signal_map_7, self.root_map

    def get_signals(self):
        self.signal_map_1 = visualize_featuremap(self.signal_map_1)
        self.signal_map_3 = visualize_featuremap(self.signal_map_3)
        self.signal_map_5 = visualize_featuremap(self.signal_map_5)
        self.signal_map_7 = visualize_featuremap(self.signal_map_7)
        self.signal_map_9 = visualize_featuremap(self.signal_map_9)
        self.signal_map_11 = visualize_featuremap(self.signal_map_11)
        return (self.signal_map_1, self.signal_map_3, self.signal_map_5,
                self.signal_map_7, self.signal_map_9, self.signal_map_11)

    def forward(self, module_stack, y, class_num, model_archi, index=None):
        if index is None:
            R = torch.eye(class_num)[torch.max(y, 1)[1]].to(y.device)
        else:
            R = torch.eye(class_num)[index].to(y.device)
        R = torch.abs(R*y)
        # R = torch.abs(R)
        r_layer = None
        for i in range(len(module_stack)):
            module = module_stack.pop()
            if len(module_stack) == 0:
                if isinstance(module, nn.Linear):
                    activation = module.activation
                    R = self.backprop_dense_input(activation, module, R)
                    print('last linear')
                elif isinstance(module, nn.Conv2d):
                    activation = module.activation
                    R = self.backprop_conv_input(activation, module, R, r_layer=[F.relu])
                else:
                    raise RuntimeError(f'{type(module)} layer is invalid initial layer type')
            elif isinstance(module, BasicBlock):
                R, r_layer = self.basic_block_R_calculate(module, R, r_layer=r_layer)
            elif isinstance(module, Bottleneck):
                print('bottleneck...')
                R = self.bottleneck_R_calculate(module, R)
            else:
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    if model_archi == 'vgg':
                        R = R.view(R.size(0), -1, 7, 7)
                        continue
                    elif model_archi == 'resnet':
                        R = R.view(R.size(0), R.size(1), 1, 1)
                activation = module.activation
                R, r_layer = self.R_calculate(activation, module, R, r_layer)
        return R

    def basic_block_R_calculate(self, basic_block, R, r_layer=None):
        if basic_block.downsample is not None:
            identity = basic_block.identity
        else:
            identity = basic_block.conv1.activation

        activation = basic_block.activation
        (R0, R1), r_layer = self.backprop_skip_connect(activation, identity, R, r_layer)
        R0, r_layer = self.backprop_conv(basic_block.conv2.activation, basic_block.conv2, R0,
                                         layer_idx=basic_block.layer_idx * 2 + 1, r_layer=r_layer)
        R0, r_layer = self.backprop_conv(basic_block.conv1.activation, basic_block.conv1, R0,
                                         layer_idx=basic_block.layer_idx * 2, r_layer=r_layer)
        if basic_block.downsample is not None:
            for i in range(len(basic_block.downsample) - 1, -1, -1):
                R1, r_layer = self.R_calculate(basic_block.downsample[i].activation,
                                               basic_block.downsample[i], R1, r_layer)
        else:
            pass
        R, r_layer = self.backprop_divide(R0, R1, r_layer)
        return R, r_layer

    def bottleneck_R_calculate(self, bottleneck, R):
        print('bottleneck...')
        if bottleneck.downsample is not None:
            identity = bottleneck.identity
        else:
            identity = bottleneck.conv1.activation
        activation = bottleneck.activation
        R0, R1 = self.backprop_skip_connect(activation, identity, R)
        R0 = self.backprop_conv(bottleneck.conv3.activation, bottleneck.conv3, R0)
        R0 = self.backprop_conv(bottleneck.conv2.activation, bottleneck.conv2, R0)
        R0 = self.backprop_conv(bottleneck.conv1.activation, bottleneck.conv1, R0)
        if bottleneck.downsample is not None:
            for i in range(len(bottleneck.downsample) - 1, -1, -1):
                R1 = self.R_calculate(bottleneck.downsample[i].activation,
                                      bottleneck.downsample[i], R1)
        else:
            pass
        R = self.backprop_divide(R0, R1)
        return R

    def R_calculate(self, activation, module, R, r_layer=None):
        if isinstance(module, nn.Linear):
            # print('linear')
            R, r_layer = self.backprop_dense(activation, module, R, r_layer=r_layer)
            return R, r_layer
        elif isinstance(module, nn.Conv2d):
            R, r_layer = self.backprop_conv(activation, module, R, r_layer=r_layer)
            return R, r_layer
        elif isinstance(module, nn.BatchNorm2d):
            R, r_layer = self.backprop_bn(R, ups=r_layer)
            return R, r_layer
        elif isinstance(module, nn.ReLU):
            R, r_layer = self.backprop_relu(activation, R, r_layer)
            return R, r_layer
        elif isinstance(module, nn.MaxPool2d):
            R, r_layer = self.backprop_max_pool(activation, module, R, r_layer)
            return R, r_layer
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            R, r_layer = self.backprop_adap_avg_pool(activation, R, r_layer)
            return R, r_layer
        elif isinstance(module, nn.Dropout):
            R, r_layer = self.backprop_dropout(R, r_layer)
            return R, r_layer
        else:
            raise RuntimeError(f"{type(module)} can not handled currently")

    def backprop_dense_input(self, activation, module, R):
        W_L = torch.clamp(module.weight, min=0)
        W_H = torch.clamp(module.weight, max=0)

        L = torch.ones_like(activation, dtype=activation.dtype) * self.lowest
        H = torch.ones_like(activation, dtype=activation.dtype) * self.highest

        Z_O = torch.mm(activation, torch.transpose(module.weight, 0, 1))
        Z_L = torch.mm(activation, torch.transpose(W_L, 0, 1))
        Z_H = torch.mm(activation, torch.transpose(W_H, 0, 1))

        Z = Z_O - Z_L - Z_H + 1e-9

        C_0 = dtd_linear_piecewise_sample(x=activation, w=module.weight, z=Z, under_R=Z, R=R, func=F.linear)
        C_L = dtd_linear_piecewise_sample(x=L, w=W_L, z=Z_L, under_R=Z_L, R=R, func=F.linear)
        C_H = dtd_linear_piecewise_sample(x=H, w=W_H, z=Z_H, under_R=Z_H, R=R, func=F.linear)
        R = C_0 - C_L - C_H
        return R

# --------------------------------经常更改的部分----------------------------------------------------------
    def backprop_dense(self, activation, module, R, r_layer=None):
        wp = torch.clamp(module.weight, min=0)
        wn = torch.clamp(module.weight, max=0)

        root = rel_sup_root_1st_cnn(activation, R, the_layer=['linear', wp])
        signal = activation - root

        zp = F.linear(signal, wp)  #
        zn = F.linear(signal, wn)  #
        Rp = dtd_linear_piecewise_sample(x=signal, w=wp, z=zp, under_R=zp, R=R, root_zero=root, func=F.linear)
        Rn = dtd_linear_piecewise_sample(x=signal, w=wn, z=zn, under_R=zn, R=R, root_zero=root, func=F.linear)
        R = Rp + Rn
        return R, ['linear', module.weight]

    def backprop_conv(self, activation, module, R, layer_idx=9, r_layer=None):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        wp = torch.clamp(module.weight, min=0)
        wn = torch.clamp(module.weight, max=0)

        root_p = rel_sup_root_1st_cnn(activation, R, the_layer=['conv2d', wp, stride, padding])
        signal_p = activation - root_p

        zp = F.conv2d(signal_p, wp, stride=stride, padding=padding)
        zn = F.conv2d(signal_p, wn, stride=stride, padding=padding)

        cp = dtd_conv_piecewise_sample(signal_p, wp, stride, padding, z=zp, under_R=zp, R=R, root_zero=root_p)
        cn = dtd_conv_piecewise_sample(signal_p, wn, stride, padding, z=zn, under_R=zn, R=R, root_zero=root_p)
        R = cp + cn

        if layer_idx == 8:
            self.signal_map_7 = R
            self.root_map = root_p
        if layer_idx == 1:
            self.signal_map_1 = R
        if layer_idx == 3:
            self.signal_map_3 = R
        if layer_idx == 5:
            self.signal_map_5 = R
        if layer_idx == 7:
            self.signal_map_7 = R
        if layer_idx == 9:
            self.signal_map_9 = R
        if layer_idx == 11:
            self.signal_map_11 = R
        return R, ['conv2d', module.stride, stride, padding]

    def backprop_conv_input(self, x, module, R, r_layer=None):
        stride, padding, kernel = module.stride, module.padding, module.kernel_size
        wp = torch.clamp(module.weight, min=0)
        # wn = torch.clamp(module.weight, max=0)
        x = torch.ones_like(x, dtype=x.dtype, requires_grad=True)

        root = rel_sup_root_1st_cnn(x, R, the_layer=['conv2d', wp, stride, padding])
        signal = x - root

        zp = F.conv2d(signal, wp, stride=stride, padding=padding)
        # zn = F.conv2d(signal, wn, stride=stride, padding=padding)

        Rp = dtd_conv_piecewise_sample(signal, wp, stride, padding, z=zp, under_R=zp, R=R, root_zero=root)
        # Rn = dtd_conv_piecewise_sample(signal, wn, stride, padding, z=zn, under_R=zn, R=R, root_zero=root)
        R = Rp
        return R

    def backprop_bn(self, R, ups=None):
        return R, ['bn']

    def backprop_dropout(self, R, ups=None):
        return R, ['drop']

    def backprop_relu(self, activation, R, r_layer=None):
        # root = rel_sup_root_1st_cnn(activation, R, the_layer=['relu'])
        # signal = activation - root
        # Z = F.relu(signal)  #
        # R = dtd_act_piecewise_sample(signal, Z, Z, R, root_zero=root, func=F.relu, step=50)
        return R, ['relu']

    def backprop_adap_avg_pool(self, activation, R, r_layer=None):
        kernel_size = activation.shape[-2:]
        Z = F.avg_pool2d(activation, kernel_size=kernel_size) * kernel_size[0] ** 2 + 1e-9
        S = R / Z
        R = activation * S
        return R, ['avgpool', kernel_size]

    def backprop_max_pool(sef, activation, module, R, ups=None):
        kernel_size, stride, padding = module.kernel_size, module.stride, module.padding
        Z, indices = F.max_pool2d(activation, kernel_size=kernel_size, stride=stride, \
                                  padding=padding, return_indices=True)
        Z = Z + 1e-9
        S = R / Z
        C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, \
                           padding=padding, output_size=activation.shape)
        R = activation * C
        return R, ['maxpool']

    def backprop_divide(self, R0, R1, ups=None):
        return R0 + R1, ['divide']

    def backprop_skip_connect(self, activation0, activation1, R, ups=None):
        Z = activation0 + activation1 + 1e-9
        S = R / Z
        R0 = activation0 * S
        R1 = activation1 * S
        return (R0, R1), ['skip']
