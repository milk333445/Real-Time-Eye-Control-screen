import torch
import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor) # 這樣可以確保v是divisor的倍數
    if new_v < 0.9 * v: # 如果new_v太小，就加上divisor
        new_v += divisor
    return new_v

def Conv2dNormActivation(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, norm_layer=None, activation_layer=None):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
    if norm_layer is not None:
        layers.append(norm_layer(out_channels))
    if activation_layer is not None:
        layers.append(activation_layer(inplace=True))
    return nn.Sequential(*layers)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
                )
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, norm_layer=None):
        super(MobileNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        last_channel = _make_divisible(1280 * width_mult, round_nearest)
        
        features = [
            Conv2dNormActivation(
                3, input_channel, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)
                )
                input_channel = output_channel
        features.append(Conv2dNormActivation(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        self.features = nn.Sequential(*features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        self._initialize_weights()
        
    def _initialize_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)   
        
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3]) # global average pooling
        x = self.classifier(x)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#model = MobileNetV2(num_classes=1)
#input_tensor = torch.randn(1, 3, 448, 448)
#output = model(input_tensor)
#print(output.shape) # torch.Size([1, 1000])
#print(count_parameters(model)) # 2225153