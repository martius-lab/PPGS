from itertools import chain
import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


def make_mlp_block(in_units, out_units, bn):
    """
    Builds a simple dense block with ReLU activation and (optional) Batch Normalization.
    """
    return [nn.Linear(in_units, out_units), nn.LayerNorm(out_units), nn.ReLU()] if bn else [nn.Linear(in_units, out_units), nn.ReLU()]


class ResnetBuilder(object):
    """
    Simplification of ResNet._make_layer(). Needs to be called sequentially.
    """

    def __init__(self, last_channel):
        self.last_channel = last_channel

    def build_layer(self, ch_out, n_blocks):
        downsample = None
        if self.last_channel != ch_out:
            downsample = nn.Sequential(
                nn.Conv2d(self.last_channel, ch_out, kernel_size=1),
                nn.BatchNorm2d(ch_out),
            )
        layers = [BasicBlock(self.last_channel, ch_out, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(ch_out, ch_out))
        self.last_channel = ch_out
        return nn.Sequential(*layers)


class Encoder(torch.nn.Module):
    """
    Model that maps batches of input states (bs x 64 x 64 x 5) composed of 3 RGB layers and 2 positional layers to
    embeddings. The embeddings can be on a latent space R^n with n=embedding_size or on an hypersphere if normalize=True.
    """
    def __init__(self, input_shape, filters, embedding_size, normalize, **kwargs):
        super(Encoder, self).__init__()

        self.first_layer = nn.Sequential(  # first layer has to adapt to the input shape of the data
            nn.Conv2d(in_channels=input_shape[0], out_channels=filters[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        builder = ResnetBuilder(last_channel=filters[0])
        self.residual_layers = nn.Sequential(*[builder.build_layer(f, 2) for f in filters])
        self.last_layer = nn.Sequential(  # last layer flattens the output of convolutions to extract an embedding
            nn.Flatten(),
            nn.Linear(self._get_linear_input_size(input_shape), embedding_size)
        )
        self.normalize = torch.nn.functional.normalize if normalize else lambda x: x
        self.is_frozen = False

    def _get_linear_input_size(self, shape):
        """
        Helper function to retrieve the input size for the final dense layer.
        """
        x = self.first_layer(torch.rand(1, *shape))
        x = self.residual_layers(x)
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.residual_layers(x)
        x = self.last_layer(x)
        return self.normalize(x)


class ForwardModel(torch.nn.Module):
    """
    Model that takes as input a batch of embeddings and a batch of actions. It recovers a representation for the latter
    and concatenates it to the state embeddings. It then feeds the resulting batch in a simple 2 layer dense neural
    network.
    """
    def __init__(self, embedding_size, action_space_size, normalize, input_shape, filters, device, **kwargs):
        super(ForwardModel, self).__init__()
        self.embedding_size = embedding_size
        self.action_space_size = action_space_size
        self.input_shape = input_shape
        self.device = device
        self.project_state = nn.Sequential(*make_mlp_block(embedding_size, 512, False))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512//4, out_channels=256, kernel_size=4, stride=2),
            nn.LayerNorm([6, 6]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2),
            nn.LayerNorm([14, 14]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.LayerNorm([31, 31]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.first_layer = nn.Sequential(  # first layer has to adapt to the input shape of the data
            nn.Conv2d(in_channels=input_shape[0]+action_space_size+1, out_channels=filters[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        builder = ResnetBuilder(last_channel=filters[0])
        self.residual_layers = nn.Sequential(*[builder.build_layer(f, 2) for f in filters])
        self.last_layer = nn.Sequential(  # last layer flattens the output of convolutions to extract an embedding
            nn.Flatten(),
            nn.Linear(self._get_linear_input_size(input_shape), embedding_size)
        )
        self.normalize = torch.nn.functional.normalize if normalize else lambda x: x
        self.is_frozen = False

    def _get_linear_input_size(self, shape):
        """
        Helper function to retrieve the input size for the final dense layer.
        """
        x = self.first_layer(torch.rand(1, shape[0]+self.action_space_size+1, *shape[1:]))
        x = self.residual_layers(x)
        return x.data.view(1, -1).size(1)

    def forward(self, z, s, a):
        a_emb = torch.eye(self.action_space_size, device=self.device)[a.squeeze().long()]  # one hot embedding
        a_emb = a_emb.reshape((-1, self.action_space_size, 1, 1)).expand((-1, self.action_space_size, 64, 64))
        z = self.project_state(z).reshape(z.shape[0], -1, 2, 2)
        z_emb = self.deconv(z)
        x = torch.cat([a_emb, z_emb, s], dim=-3)
        x = self.first_layer(x)
        x = self.residual_layers(x)
        x = self.last_layer(x)
        return self.normalize(x)


class LatentForwardModel(torch.nn.Module):
    """
    Forward model that does not rely on observations for considering the structure.
    """
    def __init__(self, embedding_size, action_space_size, normalize, forward_layers, forward_units, forward_ln, **kwargs):
        super(LatentForwardModel, self).__init__()
        self.embedding_size = embedding_size
        self.action_space_size = action_space_size
        self.layers = forward_layers
        self.units = forward_units
        self.ln = forward_ln
        self.action_lookup = nn.Embedding(self.action_space_size, self.embedding_size)
        if self.layers < 2:
            self.dense = nn.Sequential(*make_mlp_block(2*self.embedding_size, self.action_space_size, self.ln))
        else:
            self.dense = nn.Sequential(*(
                    make_mlp_block(2 * self.embedding_size, self.units, self.ln) +
                    list(chain.from_iterable(make_mlp_block(self.units, self.units, self.ln) for _ in range(self.layers-2))) +
                    [nn.Linear(self.units, self.embedding_size)]))
        self.normalize = torch.nn.functional.normalize if normalize else lambda x: x
        self.is_frozen = False

    def forward(self, z, s, a):
        a_emb = self.action_lookup(a)
        x = torch.cat([z, a_emb], -1)
        x = self.dense(x)
        return self.normalize(x)


class InverseModel(torch.nn.Module):
    """
    Inverse model mapping a pair of state embeddings to the action that takes from the first one to the second one.
    """
    def __init__(self, embedding_size, action_space_size, inverse_layers, inverse_units, inverse_ln, **kwargs):
        super(InverseModel, self).__init__()
        self.embedding_size = embedding_size
        self.action_space_size = action_space_size
        self.layers = inverse_layers
        self.units = inverse_units
        self.ln = inverse_ln
        if self.layers < 2:
            self.dense = nn.Sequential(*make_mlp_block(2*self.embedding_size, self.action_space_size, self.ln))
        else:
            self.dense = nn.Sequential(*(
                    make_mlp_block(2 * self.embedding_size, self.units, self.ln) +
                    list(chain.from_iterable(make_mlp_block(self.units, self.units, self.ln) for _ in range(self.layers-2))) +
                    [nn.Linear(self.units, self.action_space_size)]))
        self.is_frozen = False

    def forward(self, start_state, target_state):
        x = torch.cat([start_state, target_state], -1)
        return self.dense(x)


class Decoder(torch.nn.Module):

    def __init__(self, embedding_size, input_shape, **kwargs):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_size // 4, out_channels=256, kernel_size=4, stride=2),
            nn.LayerNorm([6, 6]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2),
            nn.LayerNorm([14, 14]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.LayerNorm([31, 31]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=input_shape[0], kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        self.is_frozen = False

    def forward(self, z):
        batch_size = z.shape[0]
        return self.decoder(z.reshape((batch_size, -1, 2, 2)))


class AEEncoder(Encoder):

    def __init__(self, **kwargs):
        super(AEEncoder, self).__init__(**kwargs)
        self.is_frozen = False

    def forward(self, x):
        x = self.first_layer(x)
        x = self.residual_layers(x)
        return self.last_layer(x)   # skip normalization


class VAEEncoder(Encoder):
    def __init__(self, embedding_size, input_shape, **kwargs):
        super(VAEEncoder, self).__init__(embedding_size=embedding_size, input_shape=input_shape, **kwargs)
        self.embedding_size = embedding_size
        self.last_layer = nn.Sequential(  # last layer flattens the output of convolutions to extract an embedding
            nn.Flatten(),
            nn.Linear(self._get_linear_input_size(input_shape), embedding_size * 2)
        )
        self.is_frozen = False

    def forward(self, x):
        x = self.first_layer(x)
        x = self.residual_layers(x)
        x = self.last_layer(x)
        # skip normalization
        return x[:, :self.embedding_size], torch.nn.functional.softplus(x[:, self.embedding_size:])


class VAEEncoderWrapper(nn.Module):

    def __init__(self, m):
        super(VAEEncoderWrapper, self).__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward(x)[0]


def create_component(name, args):
    if name == 'encoder':
        return Encoder(**args)
    if name == 'forward':
        return ForwardModel(**args)
    if name == 'latent_forward':
        return LatentForwardModel(**args)
    if name == 'inverse':
        return InverseModel(**args)
    if name == 'decoder':
        return Decoder(**args)
    if name == 'vae_encoder':
        return VAEEncoder(**args)
    if name == 'ae_encoder':
        return AEEncoder(**args)

    raise Exception('Unknown model name.')