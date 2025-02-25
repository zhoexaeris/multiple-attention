import os
import json
import random

# Define dataset roots
dfdcroot = 'datasets/dfdc/'
celebroot = 'datasets/celebDF/'
ffpproot = r'D:\.THESIS\datasets\3_altered\3_1_blurring'
deeperforensics_root = 'datasets/deeper/'

def load_json(name):
    """ Load JSON metadata file """
    if not os.path.exists(name):
        print(f"[ERROR] Metadata file not found: {name}")
        return []
    with open(name) as f:
        return json.load(f)

def FF_dataset(tag='DeepFakeDetection', codec='c23', part='train'):
    """
    Loads the FF++ dataset based on the new structure.

    Args:
        tag (str): Type of dataset (Origin, DeepFakeDetection, Deepfakes, etc.).
        codec (str): Compression level (c0, c23, c40, all).
        part (str): Dataset split (train, val, test, all).

    Returns:
        list: List of file paths and their labels.
    """
    assert tag in ['DeepFakeDetection', 'Deepfakes', 'FaceSwap', 'FaceShifter', 'NeuralTextures', 'Face2Face', 'Origin']
    assert codec in ['c0', 'c23', 'c40', 'all']
    assert part in ['train', 'val', 'test', 'all']

    if part == "all":
        return FF_dataset(tag, codec, 'train') + FF_dataset(tag, codec, 'val') + FF_dataset(tag, codec, 'test')

    if codec == 'all':
        return FF_dataset(tag, 'c0', part) + FF_dataset(tag, 'c23', part) + FF_dataset(tag, 'c40', part)

    print(f"[DEBUG] Looking for dataset: tag={tag}, codec={codec}, part={part}")

    # ✅ Handle REAL images (Origin)
    if tag == 'Origin':
        path_actors = os.path.join(ffpproot, '3_1_1_original', 'Actors')
        path_youtube = os.path.join(ffpproot, '3_1_1_original', 'Youtube')

        if not os.path.exists(path_actors) or not os.path.exists(path_youtube):
            raise Exception(f"[ERROR] Original dataset paths missing: {path_actors} or {path_youtube}")

        metafile_path = os.path.join(ffpproot, f"{part}.json")
        if not os.path.exists(metafile_path):
            raise Exception(f"[ERROR] Metadata file missing: {metafile_path}")

        metafile = load_json(metafile_path)
        files = []

        for i in metafile:
            actor_path = os.path.join(path_actors, i[0])
            youtube_path = os.path.join(path_youtube, i[1])

            if os.path.exists(actor_path):
                files.append([actor_path, 0])
            else:
                print(f"[WARNING] Missing actor file: {actor_path}")

            if os.path.exists(youtube_path):
                files.append([youtube_path, 0])
            else:
                print(f"[WARNING] Missing youtube file: {youtube_path}")

        return files

    # ✅ Handle FORGED images (Preprocessed)
    else:
        path_fake = os.path.join(ffpproot, '3_1_2_preprocessed', tag)

        if not os.path.exists(path_fake):
            raise Exception(f"[ERROR] Preprocessed dataset path missing: {path_fake}")

        metafile_path = os.path.join(ffpproot, f"{part}.json")
        if not os.path.exists(metafile_path):
            raise Exception(f"[ERROR] Metadata file missing: {metafile_path}")

        metafile = load_json(metafile_path)
        files = []

        for i in metafile:
            fake_img1 = os.path.join(path_fake, i[0])
            fake_img2 = os.path.join(path_fake, i[1])

            if os.path.exists(fake_img1):
                files.append([fake_img1, 1])
            else:
                print(f"[WARNING] Missing fake file: {fake_img1}")

            if os.path.exists(fake_img2):
                files.append([fake_img2, 1])
            else:
                print(f"[WARNING] Missing fake file: {fake_img2}")

        return files

# ✅ Load CelebDF Dataset
if os.path.exists(celebroot + 'celeb.json'):
    Celeb_test = list(map(lambda x: [os.path.join(celebroot, x[0]), 1 - x[1]], load_json(celebroot + 'celeb.json')))
else:
    Celeb_test = []

def make_balance(data):
    """ Balance real and fake samples """
    tr = list(filter(lambda x: x[1] == 0, data))
    tf = list(filter(lambda x: x[1] == 1, data))
    if len(tr) > len(tf):
        tr, tf = tf, tr
    rate = len(tf) // len(tr)
    res = len(tf) - rate * len(tr)
    tr = tr * rate + random.sample(tr, res)
    return tr + tf

def dfdc_dataset(part='train'):
    """ Load DFDC dataset """
    assert part in ['train', 'val', 'test']
    lf = load_json(dfdcroot + 'DFDC.json')

    if part == 'train':
        path = dfdcroot + 'dfdc/'
        files = make_balance(lf['train'])
    elif part == 'test':
        path = dfdcroot + 'dfdc-test/'
        files = lf['test']
    elif part == 'val':
        path = dfdcroot + 'dfdc-val/'
        files = lf['val']

    return [[os.path.join(path, i[0]), i[1]] for i in files]

def deeperforensics_dataset(part='train'):
    """ Load DeeperForensics dataset """
    a = os.listdir(deeperforensics_root)
    d = dict()
    for i in a:
        d[i.split('_')[0]] = i

    l = lambda x: [os.path.join(deeperforensics_root, d[x]), 1]
    metafile = load_json(os.path.join(ffpproot, f"{part}.json"))
    files = []

    for i in metafile:
        files.append(l(i[0]))
        files.append(l(i[1]))

    return files
