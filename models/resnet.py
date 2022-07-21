from typing import Any, List

from torch import nn, optim
from torch.nn import functional as F

from models import BaseConvNet
from models.baseconvnet import Tensor


class ResNet(BaseConvNet):
    def __init__(
            self,
            in_channels: int,
            out_dim: int,
            hidden_dims: List = None,
            params: dict = None,
            **kwargs
    ) -> None:
        super(ResNet, self).__init__()
        self.curr_device = None
        self.hold_graph = False
        self.params = params
        self.save_hyperparameters(self.params)

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except KeyError:
            pass

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.convnet = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1] * 3 * 4, out_dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        feat_map = self.convnet(x)
        out = self.final_layer(feat_map)
        return [feat_map, out]

    def loss_function(self, *args: Any, **kwargs) -> dict:
        prediction = args[0]
        label = args[1]
        loss = F.mse_loss(prediction, label)
        return {'loss': loss}

    def training_step(self, batch, batch_idx, **kwargs):
        real_img, label = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=label)
        loss = self.loss_function(*[results[0], label])
        self.log_dict({key: val.item() for key, val in loss.items()})
        return loss['loss']

    def validation_step(self, batch, batch_idx, **kwargs):
        real_img, label = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=label)
        loss = self.loss_function(*[results[0], label])
        self.log_dict({"val_{}".format(key): val.item() for key, val in loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.params['weight_decay']
        )
        optims.append(optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma=self.params['scheduler_gamma']
                )
                scheds.append(scheduler)
            return optims, scheds
        except KeyError:
            pass
        return optims
