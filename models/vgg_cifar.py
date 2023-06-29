import torch.nn as nn

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# cfg_io  = {
#     'vgg16':[(3,64),(64,64), (64,64), 'M', (64,128), (128,128), 'M', (128,256), (256,256), (256,256), 'M', (256,512), (512,512), (512,512), 'M', (512,512), (512,512), (512,512), 'M']
# }
cfg_in = {
    'vgg16': [3, 64, 'M', 64, 128, 'M', 128, 256, 256, 'M', 256, 512, 512, 'M', 512, 512, 512, 'M']
}


# class VGG(nn.Module):
#     def __init__(self, vgg_name, num_classes=10):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1,bias=False),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1,bias=True),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#maskvgg: perserve_ratio ratio is 1-d array 
class MaskVGG(nn.Module):
    def __init__(self, vgg_name, perserve_ratio):
        super(MaskVGG, self).__init__()
        self.perserve_ratio = perserve_ratio
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512*perserve_ratio[-1]), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.perserve_ratio[x_index - Mlayers])
                
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1,bias=True),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


#maskvgg: perserve_ratio ratio is 1-d array 
class MaskVGG_IN(nn.Module):
    def __init__(self, vgg_name, perserve_ratio):
        super(MaskVGG_IN, self).__init__()
        self.perserve_ratio = perserve_ratio
        self.features = self._make_layers(cfg_in[vgg_name])
        self.out_features = 512
        self.classifier = nn.Linear(self.out_features, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        Mlayers = 0
        y = 0
        cfg_in_channels = [i for i in cfg if i != 'M']
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.perserve_ratio[x_index - Mlayers])
                if (x_index - Mlayers) == len(self.perserve_ratio) -1:
                    y = 512
                else:
                    y = int(cfg_in_channels[x_index - Mlayers+1] * self.perserve_ratio[x_index - Mlayers+1])
                if x == 0:
                    x = 1
                if y == 0:
                    y = 1
                
                layers += [nn.Conv2d(x, y, kernel_size=3, padding=1,bias=False),
                           nn.BatchNorm2d(y),
                           nn.ReLU(inplace=True)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



#maskvgg: perserve_ratio ratio is 1-d array 
class MaskVGG_IN(nn.Module):
    def __init__(self, vgg_name, perserve_ratio, dataset_name):
        super(MaskVGG_IN, self).__init__()
        self.perserve_ratio = perserve_ratio
        self.features = self._make_layers(cfg_in[vgg_name])
        self.out_features = 512
        if "flower" in dataset_name:
            output = 102
        elif "cifar10" in dataset_name:
            output = 10
        self.classifier = nn.Linear(self.out_features, output)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        Mlayers = 0
        y = 0
        cfg_in_channels = [i for i in cfg if i != 'M']
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.perserve_ratio[x_index - Mlayers])
                if (x_index - Mlayers) == len(self.perserve_ratio) -1:
                    y = 512
                else:
                    y = int(cfg_in_channels[x_index - Mlayers+1] * self.perserve_ratio[x_index - Mlayers+1])
                if x == 0:
                    x = 1
                if y == 0:
                    y = 1
                
                layers += [nn.Conv2d(x, y, kernel_size=3, padding=1,bias=True),
                           nn.BatchNorm2d(y),
                           nn.ReLU(inplace=True)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



