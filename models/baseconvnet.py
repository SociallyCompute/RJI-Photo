from typing import List, Any, TypeVar
import pytorch_lightning as pl
from abc import abstractmethod

Tensor = TypeVar('torch.tensor')


class BaseConvNet(pl.LightningModule):

    def __init__(self) -> None:
        super(BaseConvNet, self).__init__()

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
