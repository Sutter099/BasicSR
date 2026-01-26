import random
from torch.utils import data as data
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class UnpairedDataset(data.Dataset):
    """Unpaired dataset for image restoration.
    
    Read LQ and GT images from independent folders.
    """

    def __init__(self, opt):
        super(UnpairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        
        self.paths_gt = paths_from_folder(self.gt_folder)
        self.paths_lq = paths_from_folder(self.lq_folder)
        
        self.gt_size = len(self.paths_gt)
        self.lq_size = len(self.paths_lq)
        self.dataset_size = max(self.gt_size, self.lq_size)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_index = index % self.gt_size
        gt_path = self.paths_gt[gt_index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_index = random.randint(0, self.lq_size - 1)
        lq_path = self.paths_lq[lq_index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            if gt_size:
                # Unpaired random crop: crop both images to gt_size at independent random locations
                h, w, _ = img_gt.shape
                top = random.randint(0, h - gt_size)
                left = random.randint(0, w - gt_size)
                img_gt = img_gt[top:top + gt_size, left:left + gt_size, :]
                
                h, w, _ = img_lq.shape
                top = random.randint(0, h - gt_size)
                left = random.randint(0, w - gt_size)
                img_lq = img_lq[top:top + gt_size, left:left + gt_size, :]

            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return self.dataset_size
