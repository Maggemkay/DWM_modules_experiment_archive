import torch
import torchvision
import framework
import tasks
import os
from main import register_args, load_weights, save_weights

# "Fake-run" the test command to populate 'sys.argv'
import shlex
import sys

MODEL_ID = 1
MASK_ID = 1
end_model_id = 2

while True:
    print(f"MODEL_ID: {MODEL_ID}, MASK_ID: {MASK_ID}")

    MODEL_MASK_NAME = f"repeated_pair_tests/Model-{MODEL_ID}_repeatID-{MASK_ID}"
    base_path = f"./save/{MODEL_MASK_NAME}"
    pretrained_path = f"./save/pretrained_models/model_{MODEL_ID}"

    # Note: The commands are switch depending on if a training has already occurred, and you only wish to test a pretrained model and masks.
    cmd = f"main.py -name {MODEL_MASK_NAME} -task cifar10_class_removal -stop_after 20000 -mask_loss_weight 3e-4 -mask_lr 1e-3 -step_per_mask 20000 -class_removal.keep_last_layer 1 -dropout 0.0 -cnn.dropout 0 -restore_pretrained {pretrained_path}"

    # -restore_pretrained {base_path}/model_weights"

    cmd_args = shlex.split(cmd)
    sys.argv = cmd_args

    print(cmd_args)


    #############################################

    training_helper = framework.helpers.TrainingHelper(
        wandb_project_name="modules",
        register_args=register_args, 
        extra_dirs=["export", "model_weights"])

    def invalid_task_error(self):
        assert False, f"Invalid task: {training_helper.opt.task}"

    task = tasks.Cifar10ClassRemovalTask(training_helper, pair_testing=True)


    #############################################

    assert not task.helper.opt.train_baseline
    load_weights(training_helper, task)


    if task.helper.opt.analysis.enable and not task.helper.opt.train_baseline:
        task.post_train_2()

    MASK_ID += 1

    if MASK_ID > end_model_id:
        MASK_ID = 1
        MODEL_ID += 1
    
    if MODEL_ID > end_model_id:
        print("Done with Pair-wise testing.")
        break