from sklearn.manifold import TSNE
import numpy as np
from typing import Union, List


class Reshaper(object):
    def __init__(self):
        pass

    def tsne(
            self,
            data: np.ndarray,
            ret_data: bool = True,
            **kwargs
    ) -> Union[List[Union[np.ndarray, TSNE]], TSNE]:
        t = TSNE(**kwargs)
        if ret_data:
            return [t.fit_transform(data), t]
        else:
            t.fit(data)
            return t
