import json
import torchvision.models as torch_models
from types import SimpleNamespace as Namespace
import copy
import torch
import os


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def generate_config(params, file_path):
    print("Saving Configs")
    f = open(file_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    f.write(json_data)
    f.close()


def read_config(config_path):
    # print('Parse Params file here from ', config_path, ' and pass into main')
    json_data = open(config_path, "r").read()
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))


def _get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
        if model == 'mobilenet' and dataset == 'imagenet':
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=1000)
        elif model == 'mobilenetv2' and dataset == 'imagenet':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=1000)
        elif model == 'torch_resnet18' and dataset == 'cifar10':
            net = torch_models.resnet18(pretrained=False, num_classes=10)
        elif model == 'torch_resnet50' and dataset == 'cifar10':
            net = torch_models.resnet50(pretrained=False, num_classes=10)
        elif model == 'resnet18' and dataset == 'cifar10':
            # net = torch_models.resnet18(pretrained=False, num_classes=10)
            # net.load_state_dict(torch.load('checkpoints/resnet50-19c8e357.pth'))
            # net = net.cuda()
            # if n_gpu > 1:
            #     net = torch.nn.DataParallel(net, range(n_gpu))
            # return net, copy.deepcopy(net.state_dict())
            from models.resnet import ResNet18
            net = ResNet18(num_classes=10)
        elif model == 'resnet18' and dataset == 'flower':
            # net = torch_models.resnet18(pretrained=False, num_classes=102)
            from models.resnet import ResNet18
            net = ResNet18(num_classes=102)
        elif model == 'resnet34' and dataset == 'cifar10':
            from models.resnet import ResNet34
            net = ResNet34(num_classes=10)
        elif model == 'resnet34' and dataset == 'flower':
            from models.resnet import ResNet34
            net = ResNet34(num_classes=102)
        elif model == 'resnet50' and dataset == 'cifar10':
            from models.resnet import ResNet50
            net = ResNet50(num_classes=10)
        elif model == 'resnet50' and dataset == 'flower':
            from models.resnet import ResNet50
            net = ResNet50(num_classes=102)
        elif model == 'mobilenet' and dataset == 'cifar10':
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=10)
        elif model == 'mobilenet' and 'cifar100' in dataset:
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=5)
        elif model == 'vgg16' and 'cifar100' in dataset:
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=5)
        elif model == 'mobilenet' and dataset == 'caltech101':
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=101)
        elif model == 'mobilenetv2' and dataset == 'flower':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=102)
        elif model == "vgg16" and dataset == 'cifar10':
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=10)
        elif model == "vgg16" and dataset == 'caltech101':
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=101)
        elif model == 'vgg16' and dataset == 'flower':
            from models.vgg_cifar import VGG
            net = VGG(vgg_name=model, num_classes=102)
        elif model == 'mobilenetv2' and dataset == 'cifar10':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=10)
        elif model == 'mobilenetv2' and dataset == 'caltech101':
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=101)
        elif model == 'mobilenet' and (dataset == 'MNIST' or dataset == 'FashionMNIST'):
            from models.mobilenet import MobileNetMNIST
            net = MobileNetMNIST(n_class=102) 
        elif model == "mobilenet" and dataset == "flower":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=102)
        elif model == "mobilenet" and dataset == "flower1":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=34)
        elif model == "mobilenet" and dataset == "flower2":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=34)
        elif model == "mobilenet" and dataset == "flower3":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=34)
        elif model == "mobilenetv2" and dataset == "flower1":
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=34)
        elif model == "mobilenetv2" and dataset == "flower2":
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=34)
        elif model == "mobilenetv2" and dataset == "flower3":
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=34)
        elif model == "vgg16" and dataset == "flower1":
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=34)
        elif model == "vgg16" and dataset == "flower2":
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=34)
        elif model == "vgg16" and dataset == "flower3":
            from models.vgg_cifar import VGG
            net = VGG(vgg_name="vgg16", num_classes=34)
        elif model == "mobilenet" and dataset == "flower5":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=5)
        elif model == "vgg16" and dataset == "flower5":
            from models.vgg_cifar import VGG
            net = VGG(vgg_name=model, num_classes=5)
        elif model == "mobilenetv2" and dataset == "flower5":
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=5)
        elif model == "mobilenet" and dataset == "cub200":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=200)
        elif model == "mobilenetv2" and 'cifar100' in dataset:
            from models.mobilenet_v2 import MobileNetV2
            net = MobileNetV2(n_class=5)
        elif model == "resnet18" and 'cifar100' in dataset:
            from models.resnet import ResNet18
            net = ResNet18(num_classes=5)
        else:
            raise NotImplementedError

        root_path = os.path.abspath(os.path.dirname(__file__)) + "/"
        sd = torch.load(root_path + checkpoint_path)
        if 'tar' in checkpoint_path: 
            if 'state_dict' in sd:  # a checkpoint but not a state_dict
                sd = sd['state_dict']
        else:
            sd = sd['net']
        if model == "mobilenetv2":
            res = {}
            for k, v in sd.items():
                if "classifier" in k:
                    if "weight" in k:
                        k = k.replace('classifier.0.weight', 'classifier.1.weight')
                    elif "bias" in k:
                        k = k.replace('classifier.0.bias', 'classifier.1.bias')
                res[k.replace('module.', '')] = v
            net.load_state_dict(res)
        else:
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            net.load_state_dict(sd)

        device = torch.device("cuda")
        net = net.to(device=device)
        if n_gpu > 1:
            net = torch.nn.DataParallel(net, range(n_gpu))
        return net, copy.deepcopy(net.state_dict())