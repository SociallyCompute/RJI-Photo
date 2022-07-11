from typing import List, Any, TypeVar
import pytorch_lightning as pl
from abc import abstractmethod

Tensor = TypeVar('torch.tensor')


class BaseVAE(pl.LightningModule):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, x: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
