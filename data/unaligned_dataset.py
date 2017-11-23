import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import scipy.io as sio

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot        

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform(opt)

        self.face_mask = opt.face_mask
        if self.face_mask:
            self.A_mask_dir = os.path.join(opt.dataroot, 'maskA')
            self.B_mask_dir = os.path.join(opt.dataroot, 'maskB')

            self.A_mask_paths = make_dataset(self.A_mask_dir)
            self.B_mask_paths = make_dataset(self.B_mask_dir)

            self.A_mask_paths = sorted(self.A_mask_paths)
            self.B_mask_paths = sorted(self.B_mask_paths)

            self.mask_transform = get_transform(opt, channel=1)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed)
        A = self.transform(A_img)
        if self.face_mask:
            A_mask_path = self.A_mask_paths[index_A]
            A_mask = sio.loadmat(A_mask_path)['cdata']
            random.seed(seed)
            A_mask = self.mask_transform(A_mask)
        else:
            A_mask = None

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed)
        B = self.transform(B_img)
        if self.face_mask:
            B_mask_path = self.B_mask_paths[index_B]
            B_mask = sio.loadmat(B_mask_path)['cdata']
            random.seed(seed)
            B_mask = self.mask_transform(B_mask)
        else:
            B_mask = None

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path,
                'A_mask': A_mask,  'B_mask': B_mask}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
