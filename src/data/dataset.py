import os
import numpy as np
from PIL import Image
import yaml
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.png', '.PNG', '.jpg']
TXT_EXTENSIONS = ['.txt']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)

class ImageCaptionFolderDataset(data.Dataset):
    def __init__(self, video_list,
                    img_root,
                    n_sampled_frames,
                    batch_size,
                    inversion_root,
                    inverted_img_root=None,
                    img_transform=None,
                    attribute = None,
                    crop=None, size=None, onehot=True,
                    irregular_sampling = True,
                    skip_frames = 0,
                    num_gpus=1,
                    attribute_stats = None):

        super(ImageCaptionFolderDataset, self).__init__()
        self.img_transform = img_transform
        if self.img_transform is None:
            trans_list = []
            if crop is not None:
                trans_list.append(transforms.CenterCrop(tuple(crop)))
            if size is not None:
                trans_list.append(transforms.Resize(tuple(size)))
            trans_list.append(transforms.ToTensor())
            self.img_transform = transforms.Compose(trans_list)

        self.base_seed = np.random.randint(100000)
        self.img_root = img_root

        if not isinstance(video_list, list):
            self.video_list = list(set(open(video_list).readlines()))
        else:
            self.video_list = set()
            for lst in video_list:
                self.video_list = self.video_list.union(set(open(lst).readlines()))
            self.video_list = list(self.video_list)

        self.inversion_root = inversion_root
        self.inverted_img_root = inverted_img_root
        self.n_sampled_frames = n_sampled_frames
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.attribute = attribute
        self.onehot = onehot
        self.irregular_sampling = irregular_sampling
        self.num_gpus = num_gpus
        if self.attribute is not None:
            self.attribute_stats = self.load_yaml(attribute_stats)

        self.data_paths, self.inverted_img_paths, self.inversion_paths, self.desc_paths, self.frame_numbers, self.attributes = self._load_dataset()
        print("Dataset size", len(self.data_paths))

    def _load_dataset(self):
        data_paths, inverted_img_paths, inversion_paths, desc_paths, frame_numbers, attributes = [], [], [], [], [], []
        intersection = None
        videos = []
        for idx, vid_path in enumerate(self.video_list):
            paths, im_paths, i_paths, d_paths, f_nums = {}, {}, {}, [], []
            fname = vid_path.strip()
            mn_idx = None
            for f in sorted(os.listdir(os.path.join(self.img_root, fname)))[self.skip_frames:]:
                if is_image_file(f):
                    imname = f[:-4]
                    if '_' in imname:
                        imname = imname.split('_')[-1]
                    while not imname[0].isdigit():
                        imname = imname[1:]
                    while not imname[-1].isdigit():
                        imname = imname[:-1]
                    imname = int(imname)

                    # normalize idx
                    if mn_idx is None:
                        mn_idx = imname
                        imname = 0
                    else:
                        imname -= mn_idx

                    paths[int(imname)] = os.path.join(self.img_root, fname, f)
                    if self.inverted_img_root is not None:
                        im_paths[int(imname)] = os.path.join(self.inverted_img_root, fname, f"{f[:-4]}.png")
                    f_nums.append(int(imname))

                    if self.inversion_root is not None:
                        i_paths[int(imname)] = os.path.join(os.path.join(self.inversion_root, fname, f[:-4] + ".pt"))
                elif is_text_file(f) and f == 'text_descriptions.txt': # TODO: hard code for now
                    d_paths.append(os.path.join(self.img_root, fname, f))

            if self.attribute is not None:
                att_path = os.path.join(self.img_root, fname, 'attributes.yaml')
                assert os.path.exists(att_path), f"attribute file does not exists for video {att_path}"
                attr = self.load_yaml(att_path)
                attr = attr[self.attribute]
                if self.to_onehot:
                    attr = self.to_onehot(self.attribute, attr)
                attributes.append(attr)

            data_paths.append(paths)
            inverted_img_paths.append(im_paths)
            inversion_paths.append(i_paths)
            desc_paths.append(sorted(d_paths))
            intersection = set(sorted(f_nums)) if intersection is None else intersection.intersection(set(sorted(f_nums)))
            videos.append(vid_path)
            if (idx+1) % self.batch_size == 0:
                frame_numbers.append(sorted(list(intersection)))
                intersection = None
                videos = []

        if intersection is not None:
            frame_numbers.append(sorted(list(intersection)))
        return data_paths, inverted_img_paths, inversion_paths, desc_paths, frame_numbers, attributes


    def to_onehot(self, attr, val):
        ret = np.zeros(len(self.attribute_stats[attr]))
        ret[self.attribute_stats[attr].index(val)] = 1
        return ret

    def load_yaml(self, file):
        with open(file, 'r') as f:
            return yaml.safe_load(f)

    def __getitem__(self, index):
        # adjust sampled images index accoridng to num of gpus
        index = (index // self.num_gpus) + (index % self.num_gpus) * ((len(self) // self.num_gpus) // self.batch_size) * self.batch_size
        return_list = {}

        # load text
        if index in self.desc_paths:
            rnd_txt = np.random.randint(len(self.desc_paths[index]))
            raw_desc = open(self.desc_paths[index][rnd_txt]).readlines()[0]
        else:
            raw_desc = "None"

        return_list['raw_desc'] = raw_desc


        # load attributes if available
        if self.attribute is not None:
            attribute = self.attributes[index]
            return_list['attribute'] = attribute

        # sample frames
        bin = index // self.batch_size
        local_state = np.random.RandomState(bin + self.base_seed)
        if self.irregular_sampling:
            try:
                sampleT = local_state.choice(self.frame_numbers[bin][1:], self.n_sampled_frames-1, replace=False)
                sampleT = np.append(sampleT, self.frame_numbers[bin][0])
                sampleT = np.sort(sampleT)
            except:
                print("Error: inconsistent frames indices per a batch", self.data_paths[index])
        else:
            st = local_state.randint(0, len(self.frame_numbers[bin])-self.n_sampled_frames)
            sampleT = np.array(self.frame_numbers[bin][st:st+self.n_sampled_frames])

        return_list['video_name'] = self.data_paths[index][sampleT[0]].split("/")[-2]


        I, inv_I, W = None, None, None
        for i in sampleT:
            # real image
            Ii = self.get_image(self.data_paths[index][i])
            Ii = self.img_transform(Ii)
            Ii = torch.unsqueeze(Ii, 0)
            I = Ii if I is None else torch.cat([I, Ii], dim=0)

            # inverted image
            if self.inverted_img_root is not None:
                Ii = self.get_image(self.inverted_img_paths[index][i])
                Ii = self.img_transform(Ii)
                Ii = torch.unsqueeze(Ii, 0)
                inv_I = Ii if inv_I is None else torch.cat([inv_I, Ii], dim=0)
                

            # Getting the inversion
            if self.inversion_root is not None:
                assert self.inversion_paths[index][i][len(self.inversion_root):].split('.')[0] == self.data_paths[index][i][len(self.img_root):].split('.')[0],\
                     f"inverstion path does not matched image {self.inversion_paths[index][i][len(self.inversion_root):].split('.')[0]} : {self.data_paths[index][i][len(self.img_root):].split('.')[0]}"
                w_path = self.inversion_paths[index][i]
                w_vec = self.get_inversion(w_path)
                W = w_vec if W is None else torch.cat([W, w_vec], dim=0)


        return_list['real_img']  = I
        if inv_I is not None:
            return_list['inverted_img'] = inv_I
        if W is not None:
            return_list['inversion'] = W

        return_list['sampleT'] = sampleT
        return_list['index'] = index

        return return_list

    def reset(self, ):
        self.base_seed = np.random.randint(100000)

        # permute indecies to shuffle training dataset
        self.video_list = np.random.permutation(self.video_list)
        self.data_paths, self.inverted_img_paths, self.inversion_paths, self.desc_paths, self.frame_numbers, self.attributes = self._load_dataset()

    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def get_inversion(self, inversion_path):
        w_vector = torch.load(inversion_path, map_location='cpu')
        if w_vector.shape[0] != 1:
            w_vector = w_vector.unsqueeze(0)
        # assert (w_vector.shape == (1, 18, 512)), f"Inverted vector has incorrect shape {w_vector.shape}"
        return w_vector

    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'ImageCaptionFolderDataset'
