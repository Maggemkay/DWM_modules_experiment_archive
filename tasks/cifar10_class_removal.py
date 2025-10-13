from .task import Task, Result
import dataset
from models import ConvModel
from interfaces import ConvClassifierInterface
import torch
import torchvision
import torch.utils.data
import framework
from framework.visualize import plot
from typing import Dict, Any, List, Set
from masked_model import Masks
import math

from itertools import combinations

class Cifar10ClassRemovalTask(Task):
    def __init__(self, helper, pair_testing=False):
        self._pair_testing = pair_testing   # Init the child-class before parent class
        super().__init__(helper)            # Init parent class

    def create_datasets(self):
        self.batch_dim = 0
        # self.train_set = dataset.image.CIFAR10("train")
        # self.valid_sets.iid = dataset.image.CIFAR10("valid")

        # self.train_set = dataset.image.CIFAR10("train", normalize=False)
        # self.valid_sets.iid = dataset.image.CIFAR10("valid", normalize=False)
        # scale = lambda x: (x / 255).astype("float32")

        normalize = True
        scale = lambda x: x

        self.train_set = dataset.image.CIFAR10("train", valid_split_size=0.0, normalize=normalize, augment=scale)
        self.valid_sets.iid = dataset.image.CIFAR10("test", normalize=normalize, augment=scale)
        if self._pair_testing:
            class_nums = list(range(10))
            self.mask_classes = list(combinations(class_nums, 2))
        else:
            self.mask_classes = list(range(10))

    def create_model(self):
        print("#########################")
        print("DROPOUT:")
        print(self.helper.opt.cnn.dropout)
        print("#########################")
        return ConvModel(self.train_set.in_channels(), self.train_set.out_channels(),
                         dropout=self.helper.opt.cnn.dropout)

    def create_model_interface(self):
        self.model_interface = ConvClassifierInterface(self.model)

    def get_n_masks(self) -> int:
        if self._pair_testing:
            return 1 + len(list(combinations(self.train_set.class_names, 2)))
        else:
            return 1+len(self.train_set.class_names)

    def class_removal_init_masks_and_optim(self, stage: int):
        if self.helper.opt.class_removal.keep_last_layer and stage > 0:
            names = list(sorted(self.model.masks[stage].keys()))
            last_layer_prefix = "out_layer_"

            copy_names = [n for n in names if n.startswith(last_layer_prefix)]
            optimize_names = [n for n in names if not n.startswith(last_layer_prefix)]

            print("Optimizing: ", optimize_names)

            assert len(copy_names) >= 2

            with torch.no_grad():
                for cn in copy_names:
                    self.model.masks[stage][cn].copy_(self.model.masks[0][cn])

            params = [self.model.masks[stage][n] for n in optimize_names]
        else:
            params = self.model.masks[stage].parameters()

        self.set_optimizer(torch.optim.Adam(params, self.get_mask_lr()))

    def draw_confusion_heatmap(self, hm: torch.Tensor) -> framework.visualize.plot.Heatmap:
        return plot.Heatmap(hm, "predicted", "real", round_decimals=2, x_marks=self.train_set.class_names,
                     y_marks=self.train_set.class_names)

    def plot(self, res: Result) -> Dict[str, Any]:
        # Disable periodically plotting the confusion matrices because matplotlib is very slow for big ones
        plots = super().plot(res)
        return {k: v for k, v in plots.items() if not k.endswith("/confusion")}

    def log_confusion_matrices(self):
        self.helper.summary.log({f"validation/{k}": v for k, v in self.validate().items() if k.endswith("/confusion")})

    def create_restricted_train_set(self, restrict: List[int]) -> torch.utils.data.Dataset:
        return type(self.train_set)("train", restrict=restrict)

    def post_train(self):
        self.log_confusion_matrices()
        self.prepare_model_for_analysis()

        for stage, mask_id in enumerate([-1]+self.mask_classes):

            print(f"stage / mask_id: {stage} / {mask_id}")

            split = "baseline" if stage==0 else self.train_set.class_names[mask_id]

            start = self.helper.state.iter

            self.mask_grad_norm.clear()
            self.model.set_active(stage)
            self.class_removal_init_masks_and_optim(stage)

            include_classes = [i for i in range(self.train_set.n_classes) if i != mask_id]
            print(f"Set: {include_classes}")
            set = self.create_restricted_train_set(include_classes)
            self.create_validate_on_train(set)
            loader = self.create_train_loader(set, 1234)

            for d in loader:
                # If mask is finished training
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
                    confusion = test.confusion.type(torch.float32)
                    confusion = (confusion / confusion.sum(dim=0, keepdim=True)).transpose(1, 0)
                    # Have a reference mask which is all the classes?
                    if stage == 0:
                        confusion_ref = confusion
                        self.export_tensor("class_removal/confusion_reference", confusion_ref)
                        log = {"class_removal/confusion_reference": self.draw_confusion_heatmap(confusion_ref)}
                        log.update(self.do_half_mask_test(0, "control"))
                    else:
                        diff = confusion - confusion_ref
                        log_name = f"class_removal/confusion_difference/{split}"
                        self.export_tensor(log_name, diff)
                        log = {log_name: self.draw_confusion_heatmap(diff)}
                        log.update({f"class_removal/mask_remaining/{split}/{k}": v for k, v in
                                    self.plot_remaining_stat(0, [stage]).items()})

                    self.helper.summary.log(log)
                    self.export_masks(stage)
                    break

                if self.helper.state.iter % 500 == 0:
                    print(f"Now mask batch: {self.helper.state.iter - start} (iter: {self.helper.state.iter}, start: {start})")

                res = self.train_step(d)

                plots = self.plot(res)
                plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 1000 == 0 and self.helper.opt.analysis.plot_masks:
                    plots.update({f"class_removal_masks/{split}/{k}": v for k, v in
                                  self.plot_selected_masks([0, stage] if stage > 0 else [0]).items()})

                self.helper.summary.log(plots)

    def get_half_mask_masked_layer_names(self, masks: Masks) -> List[Set[str]]:
        names = list(sorted(masks.keys()))

        feature_index = list(sorted([int(o.split("_")[1]) for o in names if o.startswith("features_") and
                                     o.endswith("_weight")]))

        n_out = sum(o.startswith("out_") and o.endswith("_weight") for o in names)
        n_keep = math.ceil((len(feature_index) + n_out)/2)
        assert n_keep < len(feature_index)

        feature_index = feature_index[:n_keep]

        return [set(sum([[n for n in names if n.startswith(f"features_{i}_")] for i in feature_index], []))]


    def post_train_2(self):
        # self.log_confusion_matrices()
        self.prepare_model_for_analysis()

        _dumb_check = True

        _i = 0

        for stage, mask_id in enumerate([-1]+self.mask_classes):

            print(f"stage / mask_id: {stage} / {mask_id}")

            if self._pair_testing:
                if stage==0:
                    split = "baseline" 
                else:
                    id1, id2 = mask_id
                    split = f"{self.train_set.class_names[id1][:4]}_{self.train_set.class_names[id2][:4]}"
            else:
                split = "baseline" if stage==0 else self.train_set.class_names[mask_id]
            

            start = self.helper.state.iter

            self.mask_grad_norm.clear()
            self.model.set_active(stage)
            self.class_removal_init_masks_and_optim(stage)

            # Include all first time to get a reference mask.
            if _dumb_check: 
                _dumb_check = False
                include_classes = [i for i in range(self.train_set.n_classes) if i != mask_id]
                set = self.create_restricted_train_set(include_classes)
                print(f"Set: {include_classes} ----> Mask class: {mask_id}")
            else:
                if self._pair_testing:
                    include_classes = list(mask_id)
                    set = self.create_restricted_train_set(include_classes)
                    print(f"Set: {include_classes} ----> Mask class: {mask_id}")
                else:
                    # include_classes = [mask_id, (mask_id + 1) % self.train_set.n_classes]
                    include_classes = [i for i in range(self.train_set.n_classes)]
                    set = self.create_restricted_train_set(include_classes)
                    print(f"Set: {include_classes} ----> Mask class: {mask_id}")

            self.create_validate_on_train(set)
            loader = self.create_train_loader(set, 1234)

            for d in loader:
                # If mask is finished training
                if self.helper.state.iter - start > self.helper.opt.step_per_mask:
                    test, _ = self.validate_on(self.valid_sets.iid, self.valid_loaders.iid)
                    confusion = test.confusion.type(torch.float32)
                    confusion = (confusion / confusion.sum(dim=0, keepdim=True)).transpose(1, 0)
                    # Have a reference mask which is all the classes?
                    if stage == 0:
                        confusion_ref = confusion
                        self.export_tensor("class_removal/confusion_reference", confusion_ref)
                        log = {"class_removal/confusion_reference": self.draw_confusion_heatmap(confusion_ref)}
                        # log.update(self.do_half_mask_test(0, "control"))
                    else:
                        # diff = confusion - confusion_ref
                        log_name = f"class_removal/confusion_mask/{split}"
                        self.export_tensor(log_name, confusion)
                        log = {log_name: self.draw_confusion_heatmap(confusion)}
                        log.update({f"class_removal/mask_remaining/{split}/{k}": v for k, v in
                                    self.plot_remaining_stat(0, [stage]).items()})

                    self.helper.summary.log(log)
                    self.export_masks(stage)
                    break
   
                if self.helper.state.iter % 5000 == 0:
                    print(f"Now mask step: {self.helper.state.iter - start} (iter: {self.helper.state.iter}, start: {start})")

                res = self.train_step(d, mask_class_id=mask_id)

                plots = self.plot(res)
                plots.update({f"analyzer/{split}/{k}": v for k, v in plots.items()})

                if self.helper.state.iter % 2000 == 0 and self.helper.opt.analysis.plot_masks:
                    plots.update({f"class_removal_masks/{split}/{k}": v for k, v in
                                  self.plot_selected_masks([0, stage] if stage > 0 else [0]).items()})

                self.helper.summary.log(plots)

            if _i >= 2:
                break