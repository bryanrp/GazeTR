import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from resnet import resnet18

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        
    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6

        # Base model (ResNet18) with output channels = maps
        self.base_model = resnet18(pretrained=True, maps=maps)

        # Create the transformer encoder
        encoder_layer = TransformerEncoderLayer(maps, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(maps)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        # Final layer to map feature vector to 2 gaze values
        self.feed = nn.Linear(maps, 2)
        self.loss_op = nn.L1Loss()

    def forward(self, x_in, feature_only=False):
        # Extract features from face input using the base (ResNet) model.
        feature = self.base_model(x_in["face"])  # shape: [batch, maps, H, W]
        batch_size = feature.size(0)
        feature = feature.flatten(2)           # shape: [batch, maps, HW]
        feature = feature.permute(2, 0, 1)       # shape: [HW, batch, maps]

        # Prepend cls token
        cls = self.cls_token.repeat(1, batch_size, 1)  # shape: [1, batch, maps]
        feature = torch.cat([cls, feature], 0)           # shape: [HW+1, batch, maps]

        # Positional embeddings (assume HW+1 tokens)
        position = torch.arange(0, feature.size(0), device=feature.device).long()
        pos_feature = self.pos_embedding(position)

        # Pass through transformer encoder.
        feature = self.encoder(feature, pos_feature)  # shape: [HW+1, batch, maps]
        feature = feature.permute(1, 2, 0)              # shape: [batch, maps, HW+1]

        # Extract the CLS token (assumed to be at position 0)
        features_cls = feature[:, :, 0]  # shape: [batch, maps]

        if feature_only:
            # Return the intermediate feature vector (32-dimensional)
            return features_cls
        else:
            # Final gaze prediction: 2 values (e.g., pitch and yaw)
            gaze = self.feed(features_cls)
            return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label)
        return loss

def load_feature_extractor(model_path, device="cuda"):
    """
    Load a trained model and detach the final self.feed layer so that the model outputs
    the intermediate 32-dimensional features.
    """
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to evaluation mode

    # Option 1: Replace the feed layer with an identity function so that forward returns features.
    model.feed = nn.Identity()

    # Option 2: Alternatively, if you don't want to modify the original forward,
    # you can call model(x_in, feature_only=True).
    return model

# Example usage:
# During training:
#   model = Model().to(device)
#   # Train the model such that model(x_in) returns gaze (2 values)
#
# After training, to get feature representations:
#   feature_extractor = load_feature_extractor("trained_model.pt")
#   features = feature_extractor(x_in)  # now returns a [batch, 32] tensor.
