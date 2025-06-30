import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .base_unet import BaseUNet, ConvBlock, Bridge, UpBlockForUNetWithResNet, initialize_weights



class Resnet18_34_Unet(BaseUNet):
    DEPTH = 6

    def __init__(self,model_type='Resnet18'):
        super().__init__()
        if model_type == 'Resnet18':
            resnet = torchvision.models.resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_type == 'Resnet34':
            resnet = torchvision.models.resnet.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        up_blocks.append(UpBlockForUNetWithResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        initialize_weights(self.bridge)
        initialize_weights(self.up_blocks)
        initialize_weights(self.out)

    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Resnet18_34_Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Resnet18_34_Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = self.out(x)
        return output_feature_map, pre_pools["layer_4"]
            
if __name__=='__main__':
    model = Resnet18_34_Unet().cuda()
    num = sum([param.nelement() for param in model.parameters()])
    num_require_grad = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    #print(model)
    print("Number of parameter: %.5fM" % (num / 1e6))
    print("Number of require grad parameter: %.5fM" % (num_require_grad / 1e6))
    
    inp = torch.rand((2, 3, 512, 512)).cuda()
    out,_ = model(inp)
    print(out.shape)
    