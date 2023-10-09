import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate


class GeneralImgCapCsvDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root='', ann_paths=[], 
        img_path_key='filepath', caption_key='title'
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            df = pd.read_csv(ann_path)
            for idx in tqdm(df.index):
                img_path = df.loc[idx, img_path_key]
                caption = df.loc[idx, caption_key]
                self.annotation.append([img_path, caption])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

#         self._add_instance_ids()
        
#         self.img_ids = {}
#         n = 0
#         for ann in self.annotation:
#             img_id = ann["image_id"]
#             if img_id not in self.img_ids.keys():
#                 self.img_ids[img_id] = n
#                 n += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
#         ann = self.annotation[index]
        image_path, caption = self.annotation[index]

        image_path = os.path.join(self.vis_root, image_path)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(caption)

        return {
            "image": image,
            "text_input": caption,
            # "image_id": self.img_ids[ann["image_id"]],
        }

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

#     def _add_instance_ids(self, key="instance_id"):
#         for idx, ann in enumerate(self.annotation):
#             ann[key] = str(idx)