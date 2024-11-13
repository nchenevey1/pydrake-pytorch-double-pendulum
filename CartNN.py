

import torch.nn as nn

class CartNN(nn.Module):
    def __init__(self, layer_sizes=None, state_dict=None):
        super(CartNN, self).__init__()
        layers = []

        if state_dict is not None:
            for key in state_dict.keys():
                if 'weight' in key:
                    in_features = state_dict[key].shape[1]
                    out_features = state_dict[key].shape[0]
                    layers.append(nn.Linear(in_features, out_features))
                    layers.append(nn.Tanh())
                elif 'bias' in key:
                    if out_features is not None and out_features not in [layer.out_features for layer in layers if isinstance(layer, nn.Linear)]:
                        layers.append(nn.Linear(in_features, out_features))

            if layers:
                layers.pop()

        elif layer_sizes is not None:
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(layer_sizes[-1], 1))

        self.model = nn.Sequential(*layers)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)