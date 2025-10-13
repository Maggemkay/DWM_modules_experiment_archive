import torch
from torch.nn import functional as F
from typing import Dict
from .result import FeedforwardResult
from .model_interface import ModelInterface


class ConvClassifierInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def create_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["image"]

    # Old code
    # def loss(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     return F.cross_entropy(net_out, data["label"].long().flatten())

    def loss_CCE(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(net_out, data["label"].long().flatten())
        
    def loss_BCE(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor], mask_class_id: int = None) -> torch.Tensor:

        # define the class of interest
        cls_id = mask_class_id
        # get the prediction for the class of interest (assume [batch_size, classes])
        pred = torch.softmax(net_out, dim=1)
        # convert class labels into 0/1 labels depending on if the class is the class of interest
        true = (data["label"].long().flatten() == cls_id).float()

        # return loss
        return F.binary_cross_entropy(pred[:, cls_id], true)

    def decode_outputs(self, outputs: FeedforwardResult) -> torch.Tensor:
        return outputs.outputs.argmax(-1)

    # OLD CODE
    # def __call__(self, data: Dict[str, torch.Tensor]) -> FeedforwardResult:
    #     input = self.create_input(data)

    #     res = self.model(input)
    #     loss = self.loss(res, data)

    #     return FeedforwardResult(res, loss)

    def __call__(self, data: Dict[str, torch.Tensor], mask_class_id: int = None) -> FeedforwardResult:
        input = self.create_input(data)

        res = self.model(input)

        if isinstance(mask_class_id, (list, tuple)) and len(mask_class_id) > 1:
            loss = self.loss_CCE(res, data)
        elif mask_class_id != None and mask_class_id != -1:
            loss = self.loss_BCE(res, data, mask_class_id)
        else:
            loss = self.loss_CCE(res, data)

        return FeedforwardResult(res, loss)
