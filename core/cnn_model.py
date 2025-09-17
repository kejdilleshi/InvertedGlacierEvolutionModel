
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, nb_inputs, nb_outputs, config):
        super().__init__()
        layers = []
        in_channels = nb_inputs
        for _ in range(config['nb_layers']):
            layers.append(nn.Conv2d(in_channels, config['nb_out_filter'],
                                    kernel_size=config['conv_ker_size'], padding='same'))
            layers.append(nn.ReLU() if config['activation'] == "relu" else nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(config['dropout_rate']))
            in_channels = config['nb_out_filter']
        layers.append(nn.Conv2d(in_channels, nb_outputs, kernel_size=1))
        self.model = nn.Sequential(*layers)     

    def forward(self, x):
        return self.model(x)   