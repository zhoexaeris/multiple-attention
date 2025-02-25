from config import train_config
from train import distributed_train
from evaluation import all_eval
import argparse
import fire
import torch
import subprocess

# torch.autograd.set_detect_anomaly(True)

# Pretrain the model using a specific dataset (FF++ C23, C40, CelebDF, WildDeepfake)
def pretrain(dataset):
    name = f'Efb4_{dataset}'
    url = 'tcp://127.0.0.1:27015'

    Config = train_config(
        name,
        [dataset, 'efficientnet-b4'], 
        url=url,
        attention_layer='b5',
        feature_layer='b1',
        epochs=20,
        batch_size=64
    )

    Config.mkdirs()
    distributed_train(Config)

    # Evaluate last 3 checkpoints
    procs = [subprocess.Popen(['/bin/bash', '-c', f'CUDA_VISIBLE_DEVICES={i} python main.py test {name} {j}'])
             for i, j in enumerate(range(-3, 0))]
    
    for i in procs:
        i.wait()


# Run a custom experiment with modified parameters
def experiment():
    name = 'exp1_b5_b2'
    url = 'tcp://127.0.0.1:27016'

    # 🔹 Adjusted dataset structure for pre-extracted frames
    Config = train_config(
        name,
        ['ff-c23-c40-celebdf-wilddeepfake', 'efficientnet-b4'],
        url=url,
        attention_layer='b5',
        feature_layer='b2',
        epochs=50,
        batch_size=32,  # Reduce batch size if memory is an issue
        ckpt='checkpoints/Efb4/ckpt_19.pth',
        inner_margin=[0.2, -0.8],
        margin=0.8
    )

    Config.mkdirs()
    distributed_train(Config)

    procs = [subprocess.Popen(['/bin/bash', '-c', f'CUDA_VISIBLE_DEVICES={i} python main.py test {name} {j}'])
             for i, j in enumerate(range(-3, 0))]

    for i in procs:
        i.wait()

# Resume training from the last checkpoint
def resume(name, epochs=0):
    Config = train_config.load(name)
    Config.epochs += epochs
    Config.reload()
    Config.resume_optim = True

    distributed_train(Config)

    for i in range(-3, 0):
        all_eval(name, i)

# Run evaluation on a specific checkpoint.
def test(name, ckpt=None):
    all_eval(name, ckpt)

if __name__ == "__main__":
    fire.Fire()
