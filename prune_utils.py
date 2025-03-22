import torch

def get_layer_norm(model):
    norms = []
    nLayers = len(model.model.layers)
    for i in range(nLayers):
        layer_norm = 0
        for name, param in model.model.layers[i].named_parameters():
            if ("weight" in name) and (("self_attn" in name) or ("mlp" in name)):
                layer_norm += torch.linalg.matrix_norm(param, ord='fro') # L2 Norm

        norms.append((i, layer_norm.item()))

    return sorted(norms, key=lambda x: x[1])


def get_pruned_model(model, num_layers_to_prune=5):
    # model_pruned =copy.deepcopy(model)
    layer_norms = get_layer_norm(model)

    # Get indices of the least important layers
    layers_to_prune = [idx for idx, _ in layer_norms[:num_layers_to_prune]]

    # Rebuild the model without these layers
    new_layers = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in layers_to_prune]
    )

    model.model.layers = new_layers

    return model

def get_sensitvity_pruned_model(model, idx):
    layers_to_prune = [idx]

    # Rebuild the model without these layers
    new_layers = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in layers_to_prune]
    )

    model.model.layers = new_layers

    return model