import os
import pickle
import glob
import datetime

class train_config:
    def __init__(self, name, recipes=[], **params):
        ### Basic Configs
        self.name = name
        self.comment = ''
        self.workers = 10  # Number of data loader workers
        self.epochs = 20  # Increase epochs since frames train faster
        self.learning_rate = 3e-4  # Adjusted for frame-based training
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 1e-6
        self.scheduler_step = 5
        self.scheduler_gamma = 0.5
        self.ckpt = ''  # Load pre-trained checkpoint if available
        self.resume_optim = False
        self.freeze = []
        self.url = 'tcp://127.0.0.1:27015'

        ### Dataset Config
        self.num_classes = 2  # Binary classification (Real vs Fake)
        self.num_attentions = 4
        self.attention_layer = 'b5'
        self.feature_layer = 'b1'
        self.mid_dims = 256
        self.dropout_rate = 0.25
        self.drop_final_rate = 0.5
        self.pretrained = ''  # Path to pretrained model if used

        ### Dataset Paths (UPDATED)
        self.datalabel_c23 = 'ff-c23'
        self.datalabel_c40 = 'ff-c40'
        self.datalabel_celebdf = 'celebdf'
        self.datalabel_wild = 'wilddeepfake'
        self.imgs_per_video = None  # Not needed anymore
        self.frame_interval = None  # No need for frame skipping
        self.max_frames = None  # No need to limit frames

        ### Image Preprocessing (Updated for Frames)
        self.resize = (320, 320)  # Maintain resolution for model input
        self.normalize = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Keep normalization
        self.augment = None  # Disable augmentations

        ### Training Parameters
        self.batch_size = 64  # Increase batch size for faster training
        self.alpha = 0.05
        self.alpha_decay = 0.9
        self.margin = 0.5
        self.inner_margin = [0.1, -2]

        ### Loss Weights
        self.ensemble_loss_weight = 1
        self.aux_loss_weight = 0.5
        self.AGDA_loss_weight = 1
        self.match_loss_weight = 0.1

        ### Store Configurations
        for i in recipes:
            self.recipe(i)
        for i in params:
            self.__setattr__(i, params[i])

        self.train_dataset = dict(datalabel=self.datalabel, resize=self.resize, imgs_per_video=self.imgs_per_video, normalize=self.normalize)
        self.val_dataset = self.train_dataset

        self.net_config = dict(
            net='efficientnet-b4',
            feature_layer=self.feature_layer,
            attention_layer=self.attention_layer,
            num_classes=self.num_classes,
            M=self.num_attentions,
            mid_dims=self.mid_dims,
            dropout_rate=self.dropout_rate,
            drop_final_rate=self.drop_final_rate,
            pretrained=self.pretrained,
            alpha=self.alpha,
            size=self.resize,
            margin=self.margin,
            inner_margin=self.inner_margin
        )

    def mkdirs(self):
        """ Create directories for training logs and checkpoints """
        os.makedirs(f'checkpoints/{self.name}', exist_ok=True)
        os.makedirs(f'runs/{self.name}', exist_ok=True)
        os.makedirs(f'evaluations/{self.name}', exist_ok=True)

        with open(f'runs/{self.name}/config.pkl', 'wb') as f:
            pickle.dump(self, f)

        if not self.comment:
            self.comment = self.name + '_' + datetime.datetime.now().isoformat()

        os.system(f'git add . && git commit -m "{self.comment}" && git tag {self.name} -f')

    @staticmethod
    def load(name):
        """ Load saved training configuration """
        with open(f'runs/{name}/config.pkl', 'rb') as f:
            p = pickle.load(f)

        v = train_config('', ['ff-', 'efficientnet-b4'])
        p = vars(p)

        for i in p:
            setattr(v, i, p[i])

        return v

    def reload(self):
        """ Reload training with latest checkpoint """
        list_of_files = glob.glob(f'checkpoints/{self.name}/*')
        num = len(list_of_files)

        if num > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.ckpt = latest_file
