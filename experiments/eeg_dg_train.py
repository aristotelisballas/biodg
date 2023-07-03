import copy
import os
import time

import numpy as np
import torch
from absl import app, flags
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from EEG import hparams_registry
from EEG.config import pickle_data_dir, eeg_results_dir
from EEG.models import algorithms
# from ECG.models.resnet1d import resnet18
from commons import BioSignal
from dataset import PickleDataset
from utils import get_files, get_datafile_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define Experiment flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train the model', 0)
flags.DEFINE_string('algorithm', 'ERM', 'Method of model training')
flags.DEFINE_string('holdout', 'CHINA', 'holdout')
flags.DEFINE_string('pretrained', 'yes', 'Model pretrained or not')
flags.DEFINE_integer('features', 200, 'Model Features')
flags.DEFINE_integer('p', 4, 'Reduction Hyperparameter')
flags.DEFINE_integer('batch_size', 128, 'Reduction Hyperparameter')
flags.DEFINE_float('lr', 0.00009, 'Optimizer Learning rate')
flags.DEFINE_float('momentum', 0.9, 'Optimizer Momentum')
flags.DEFINE_float('wd', 0.0005, 'Optimizer Weight Decay')
flags.DEFINE_string('optim', 'sgd', 'Optimizer to be used')
flags.DEFINE_string('c', '', 'Comments for experiment')


def experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    algorithm = FLAGS.algorithm
    hparams = hparams_registry.default_hparams(algorithm, 'webcam')
    hparams['lr'] = FLAGS.lr
    hparams['weight_decay'] = 0.0005

    algorithm_class = algorithms.get_algorithm_class(algorithm)
    model = algorithm_class((62, 170, 5), 3, 2, hparams)
    model.to(device)

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    biosignal = BioSignal.EEG
    holdout = str(FLAGS.holdout)
    datafile_dir = get_datafile_dir(biosignal, holdout)

    train_data = get_files(pickle_data_dir, datafile_dir / 'train.txt')
    val_data = get_files(pickle_data_dir, datafile_dir / 'val.txt')
    test_data = get_files(pickle_data_dir, datafile_dir / 'test.txt')
    holdout_data = get_files(pickle_data_dir, datafile_dir / 'holdout.txt')

    pickle_datasets = {'train': PickleDataset(train_data),
                       'val': PickleDataset(val_data),
                       'test': PickleDataset(test_data),
                       'holdout': PickleDataset(holdout_data)
                       }

    dataset_sizes = {'train': pickle_datasets['train'].__len__(),
                     'val': pickle_datasets['val'].__len__(),
                     'test': pickle_datasets['test'].__len__(),
                     'holdout': pickle_datasets['holdout'].__len__()
                     }

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(22)
        torch.manual_seed(22)
    else:
        torch.manual_seed(22)

    dataloaders = {x: DataLoader(pickle_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val', 'test', 'holdout']}

    train_steps = dataset_sizes['train'] // batch_size
    val_steps = dataset_sizes['val'] // batch_size
    test_steps = dataset_sizes['test'] // batch_size
    holdout_steps = dataset_sizes['holdout'] // batch_size

    outPath = eeg_results_dir / FLAGS.holdout / FLAGS.algorithm / FLAGS.network

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
                    minibatches_device = [(inputs.float().to(device), labels.to(device))]

                    running_counter += 1
                    if phase == 'train':
                        if running_counter == train_steps:
                            print(f'Training______Batch: {running_counter} / {train_steps}')
                            print(torch.cuda.memory_allocated())
                        else:
                            print(f'Training______Batch: {running_counter} / {train_steps}', end='\r')
                    elif phase == 'val':
                        if running_counter == val_steps:
                            print(f'Validation____Batch: {running_counter} / {val_steps}')
                        else:
                            print(f'Validation____Batch: {running_counter} / {val_steps}', end='\r')

                    step_vals = model.update(minibatches_device)
                    loss = step_vals['loss']

                    # statistics
                    running_loss += loss * inputs.size(0)
                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                tb.add_scalar(f"{phase} Loss", epoch_loss, epoch)
                print(f'{phase} Loss: {epoch_loss}')

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
        training_time = 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_loss

    def eval_model(model, phase):
        model.eval()

        running_corrects_1 = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.predict(inputs.float())
                _, preds_1 = torch.max(outputs, 1)

                running_corrects_1 += torch.sum(preds_1 == labels.data)

        test_acc1 = running_corrects_1.double() / dataset_sizes[phase]

        return test_acc1

    # Decay LR by a factor of 0.1
    try:
        exp_lr_scheduler = lr_scheduler.StepLR(model.optimizer, step_size=epochs - 6, gamma=0.1)
    except:
        exp_lr_scheduler = None

    # Train and evaluate
    model, best_loss = train_model(model, exp_lr_scheduler, epochs)
    test_acc_1 = eval_model(model, phase='test')
    holdout_test_acc_1 = eval_model(model, phase='holdout')

    print(f'Inter-Domain Accuracy - {test_acc_1}')
    print(f'Holdout Domain Accuracy - {holdout_test_acc_1}')

    print(outPath)

    f = open(outPath / "details.txt", "w")
    details = f"""---------------Training Details---------------
           Configuration: Batch: {batch_size} , Epochs: {epochs}, Network: {FLAGS.network},
           Params:{hparams}, Training time: {training_time}
           ----------------Comments----------------
           Comments: {FLAGS.c}
            ---------------Evaluation---------------
           Inter-Domain: Top-1: {test_acc_1}
           Holdout: Top-1 {holdout_test_acc_1}
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
