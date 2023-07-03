import copy
import os
import time

import numpy as np
import torch
from absl import app, flags
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ECG import hparams_registry
from ECG.bioconfig import pickle_data_dir, ecg_results_dir
from ECG.models import algorithms
from commons import BioSignal
from dataset import PickleDataset
from utils import get_files, get_datafile_dir, calculate_per_class_prediction_metrics_ecg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define Experiment flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 30, 'Number of epochs to train the model', 0)
flags.DEFINE_string('algorithm', 'ERM', 'Training Algorithm')
flags.DEFINE_string('backbone', 'resnet18', 'Backbone model: sresnet or resnet18')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('lr', 0.001, 'Optimizer Learning rate')
flags.DEFINE_float('momentum', 0.9, 'Optimizer Momentum')
flags.DEFINE_float('wd', 0.0005, 'Optimizer Weight Decay')
flags.DEFINE_string('optim', 'sgd', 'Optimizer to be used')
flags.DEFINE_string('c', '', 'Comments for experiment')


def experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    algorithm = FLAGS.algorithm
    backbone = FLAGS.backbone

    hparams = hparams_registry.default_hparams(algorithm, 'webcam')
    hparams['lr'] = FLAGS.lr
    hparams['weight_decay'] = FLAGS.wd

    if backbone == 'sresnet':
        hparams['resnet18'] = False
    elif backbone == 'resnet18':
        hparams['resnet18'] = True
    else:
        raise NotImplementedError

    algorithm_class = algorithms.get_algorithm_class(algorithm)
    model = algorithm_class((12, 5000), 24, 2, hparams)
    model.to(device)

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    biosignal = BioSignal.ECG

    datafile_dir = get_datafile_dir(biosignal)

    train_data = get_files(pickle_data_dir, datafile_dir / 'train.txt')
    val_data = get_files(pickle_data_dir, datafile_dir / 'val.txt')
    test_data = get_files(pickle_data_dir, datafile_dir / 'test.txt')
    holdout_data = get_files(pickle_data_dir, datafile_dir / 'holdout.txt')

    pickle_datasets = {'train': PickleDataset(train_data, biosignal=biosignal),
                       'val': PickleDataset(val_data, biosignal=biosignal),
                       'test': PickleDataset(test_data, biosignal=biosignal),
                       'holdout': PickleDataset(holdout_data, biosignal=biosignal)
                       }

    dataset_sizes = {'train': pickle_datasets['train'].__len__(),
                     'val': pickle_datasets['val'].__len__(),
                     'test': pickle_datasets['test'].__len__(),
                     'holdout': pickle_datasets['holdout'].__len__()
                     }
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(22)
        torch.manual_seed(42)
    else:
        torch.manual_seed(22)

    dataloaders = {x: DataLoader(pickle_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val', 'test', 'holdout']}

    train_steps = dataset_sizes['train'] // batch_size
    val_steps = dataset_sizes['val'] // batch_size
    test_steps = dataset_sizes['test'] // batch_size
    holdout_steps = dataset_sizes['holdout'] // batch_size

    outPath = ecg_results_dir / FLAGS.algorithm / FLAGS.network

    experiment_num = len(np.array(sorted([item.name for item in outPath.glob('*')]))) + 1
    outPath = outPath / f'training_{experiment_num}'
    os.makedirs(str(outPath))

    def train_model(model, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 50.00

        # Tensorboard Writer
        tb = SummaryWriter(outPath)

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_counter = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    minibatches_device = [(inputs.to(device), labels.float().to(device))]

                    running_counter += 1
                    if phase == 'train':
                        if running_counter == train_steps:
                            print(f'Training______Batch: {running_counter} / {train_steps}')
                        else:
                            print(f'Training______Batch: {running_counter} / {train_steps}', end='\r')
                    elif phase == 'val':
                        if running_counter == val_steps:
                            print(f'Validation____Batch: {running_counter} / {val_steps}')
                        else:
                            print(f'Validation____Batch: {running_counter} / {val_steps}', end='\r')
                    uda_device = None
                    step_vals = model.update(minibatches_device, uda_device)
                    loss = step_vals['loss']

                    # statistics
                    running_loss += loss * inputs.size(0)

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss}')
                tb.add_scalar(f"{phase} Loss", epoch_loss, epoch)
                # deep copy the model
                if phase == 'train' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()

        time_elapsed = time.time() - since
        global training_time
        training_time = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_loss

    # Decay LR by a factor of 0.1
    try:
        exp_lr_scheduler = lr_scheduler.StepLR(model.optimizer, step_size=epochs - 6, gamma=0.1)
    except:
        exp_lr_scheduler = None

    # Train and evaluate
    model, best_loss = train_model(model, exp_lr_scheduler, epochs)

    y_true_test = []
    y_true_holdout = []
    y_hat_holdout = []
    y_hat_test = []

    print('Getting Inter Test Predictions')
    for i in range(test_steps):
        x, y = next(iter(dataloaders['test']))
        x = x.to(device)
        y = y.to(device)
        y_true_test.append(y)
        y_hat_test.append((model.predict(x) > 0.5) * 1)

    print('Getting Holdout Test Predictions')
    for i in range(holdout_steps):
        x, y = next(iter(dataloaders['holdout']))
        x = x.to(device)
        y = y.to(device)
        y_true_holdout.append(y)
        y_hat_holdout.append((model.predict(x) > 0.5) * 1)

    y_true_test = torch.cat(y_true_test)
    y_hat_test = torch.cat(y_hat_test)
    y_true_holdout = torch.cat(y_true_holdout)
    y_hat_holdout = torch.cat(y_hat_holdout)

    df_test_metrics = calculate_per_class_prediction_metrics_ecg(y_true_test, y_hat_test)
    df_test_metrics.to_csv(outPath / 'test_metrics.csv')

    df_holdout_metrics = calculate_per_class_prediction_metrics_ecg(y_true_holdout, y_hat_holdout)
    df_holdout_metrics.to_csv(outPath / 'test_holdout_metrics.csv')
    print(outPath)
    f = open(outPath / "details.txt", "w")
    details = f"""---------------Training Details---------------
           Configuration: Batch: {batch_size} , Epochs: {epochs}, Network: {FLAGS.network},
           Hparams: {hparams}, Training Time: {training_time}
           ----------------Comments----------------
           Comments: {FLAGS.c}
           ---------------Evaluation---------------
           Inter-Domain: Top-1: 
           OOD: Top-1: 
           ---------------Algorithm------------------
           Algorithm: {FLAGS.algorithm}
           ---------------Model Summary---------------
           Best Loss: {best_loss}
           -------------------------------------------         
           Model: {model}"""
    f.write(details)
    f.close()

    torch.save(model, outPath / "model.pt")


#
# ---   Main   ---
def main(args):
    del args

    experiment()


if __name__ == "__main__":
    app.run(main)
