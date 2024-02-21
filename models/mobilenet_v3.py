import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv2dNormActivation(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, norm_layer=None, activation_layer=None):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
    if norm_layer is not None:
        layers.append(norm_layer(out_channels))
    if activation_layer is not None:
        layers.append(activation_layer(inplace=True))
    return nn.Sequential(*layers)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor) # 這樣可以確保v是divisor的倍數
    if new_v < 0.9 * v: # 如果new_v太小，就加上divisor
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.Hardsigmoid(inplace=True)
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # (b, c, 1, 1) -> (b, c)
        y = self.fc(y).view(b, c, 1, 1) # (b, c) -> (b, c, 1, 1)
        return x * y
    
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, nl, se=False, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        
        if nl == "RE":
            activation_layer = nn.ReLU
        elif nl == "HS":
            activation_layer = nn.Hardswish
        else:
            raise ValueError("Unsupported nonlinearity: " + nl)
        
        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer
                )
            )
        layers.extend(
            [
                Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup)
            ]
        )
        
        if se:
            layers.insert(2, SELayer(hidden_dim)) # 在第三層插入SELayer
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)
        
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2, mode="small"):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1024 if mode == "large" else 1280
        
        if mode == "large":
            layers = [
               # k, exp, c,    se,  nl,  s
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == "small":
            layers = [
               # k, exp, c,    se,  nl,  s
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
            
        firstconv_output_channel = _make_divisible(input_channel * width_mult, 8)
        features = [
            Conv2dNormActivation(
                3, firstconv_output_channel, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish
            )
        ]
        for k ,exp, c, se, nl, s in layers:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            features.append(
                InvertedResidual(
                    input_channel, output_channel, s, expand_ratio=exp_channel//input_channel, nl=nl, se=se, norm_layer=nn.BatchNorm2d
                )
            )
            input_channel = output_channel
        lastconv_output_channel = _make_divisible(6 * width_mult, 8)
        features.append(
            Conv2dNormActivation(
                input_channel, lastconv_output_channel, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish
            )
        )
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        
    def forward(self, x):
        x = self.features(x) # (1, 1280, 7, 7)
        x = self.avgpool(x) # (1, 1280, 1, 1)
        x = torch.flatten(x, 1) # (1, 1280)
        x = self.classifier(x) # (1, 1000)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
#model = MobileNetV3(num_classes=1000, mode="small")
#x = torch.randn(1, 3, 224, 224)
#print(model(x).shape) # torch.Size([1, 1])
#print(count_parameters(model)) # 2246436
