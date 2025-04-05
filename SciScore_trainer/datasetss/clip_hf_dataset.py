from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from datasetss.base_dataset import BaseDataset, BaseDatasetConfig
import random
import itertools

logger = get_logger(__name__)

@dataclass
class ProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")


@dataclass
class CLIPHFDatasetConfig(BaseDatasetConfig):
    _target_: str = "datasetss.clip_hf_dataset.CLIPHFDataset"
    dataset_config_name: str = "null"

    from_disk: bool = False
    cache_dir: Optional[str] = None

    dataset_name: str = "Jialuo21/Science-T2I-Fullset"
    train_split_name: str = "train"
    test_split_name: str = "test_C"
    caption_column_name: str = "caption"
    pos_caption_column_name: str = "positive_caption"
    neg_caption_column_name: str = "negative_caption"
    image_pos_column_name: str = "positive_imgs"
    image_neg_column_name: str = "negative_imgs"
    category_column_name: str = "category"

    input_ids_column_name: str = "input_ids"
    pos_input_ids_column_name: str = "positive_input_ids"
    neg_input_ids_column_name: str = "negative_input_ids"
    pixels_pos_column_name: str = "pixel_values_pos"
    pixels_neg_column_name: str = "pixel_values_neg"

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

class CLIPHFDataset(BaseDataset):

    def __init__(self, cfg: CLIPHFDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        logger.info(f"Loading {self.split} dataset")

        self.dataset = self.load_hf_dataset(self.split)
        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    def select_combination(self, num_captions, num_pos_imgs, num_neg_imgs, num_pos_captions, num_neg_captions):
        caption_idx = random.randint(0, num_captions - 1)
        pos_caption_idx = random.randint(0, num_pos_captions - 1)
        neg_caption_idx = random.randint(0, num_neg_captions - 1)
        pos_img_idx = random.randint(0, num_pos_imgs - 1)
        neg_img_idx = random.randint(0, num_neg_imgs - 1)
        return caption_idx, pos_caption_idx, neg_caption_idx, pos_img_idx, neg_img_idx
        
    
    def load_hf_dataset(self, split: str) -> Dataset:
        if self.cfg.from_disk:
            dataset = load_from_disk(self.cfg.dataset_name)[split]
        else:
            dataset = load_dataset(
                self.cfg.dataset_name,
                cache_dir=self.cfg.cache_dir,
                split=split
            )
        return dataset

    def tokenize(self, example):
        def get_input_ids(text):
            return self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
        
        input_ids = get_input_ids(example[self.cfg.caption_column_name])
        pos_input_ids = get_input_ids(example[self.cfg.pos_caption_column_name])
        neg_input_ids = get_input_ids(example[self.cfg.neg_caption_column_name])
        
        return input_ids, pos_input_ids, neg_input_ids


    def process_images(self, images):
        pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
        return pixel_values

    def __getitem__(self, idx):
        example = self.dataset[idx]
        if self.split == self.cfg.train_split_name:
            num_captions = len(example[self.cfg.caption_column_name])
            num_pos_imgs = len(example[self.cfg.image_pos_column_name])
            num_neg_imgs = len(example[self.cfg.image_neg_column_name])
            num_pos_captions = len(example[self.cfg.pos_caption_column_name])
            num_neg_captions = len(example[self.cfg.neg_caption_column_name])
            caption_idx, pos_caption_idx, neg_caption_idx, pos_img_idx, neg_img_idx = self.select_combination(num_captions, num_pos_imgs, num_neg_imgs, num_pos_captions, num_neg_captions)
            
            example[self.cfg.caption_column_name] = example[self.cfg.caption_column_name][caption_idx]
            example[self.cfg.image_pos_column_name] = example[self.cfg.image_pos_column_name][pos_img_idx]
            example[self.cfg.image_neg_column_name] = example[self.cfg.image_neg_column_name][neg_img_idx]
            example[self.cfg.pos_caption_column_name] = example[self.cfg.pos_caption_column_name][pos_caption_idx]
            example[self.cfg.neg_caption_column_name] = example[self.cfg.neg_caption_column_name][neg_caption_idx]
        else:
            example[self.cfg.caption_column_name] = example[self.cfg.caption_column_name][0]
            example[self.cfg.image_pos_column_name] = example[self.cfg.image_pos_column_name][0]
            example[self.cfg.image_neg_column_name] = example[self.cfg.image_neg_column_name][0]
            
        input_ids, pos_input_ids, neg_input_ids = self.tokenize(example)

        pixel_pos_values = self.process_images(example[self.cfg.image_pos_column_name])
        pixel_neg_values = self.process_images(example[self.cfg.image_neg_column_name])
        if self.split == self.cfg.train_split_name:
            item = {
                self.cfg.input_ids_column_name: input_ids,
                self.cfg.pos_input_ids_column_name: pos_input_ids,
                self.cfg.neg_input_ids_column_name: neg_input_ids,
                self.cfg.pixels_pos_column_name: pixel_pos_values,
                self.cfg.pixels_neg_column_name: pixel_neg_values,
            }
        else:
            item = {
                self.cfg.input_ids_column_name: input_ids,
                self.cfg.pos_input_ids_column_name: pos_input_ids,
                self.cfg.neg_input_ids_column_name: neg_input_ids,
                self.cfg.pixels_pos_column_name: pixel_pos_values,
                self.cfg.pixels_neg_column_name: pixel_neg_values,
                self.cfg.category_column_name: example[self.cfg.category_column_name]
            }
        return item
    
    def simple_collate(self, batch, column_name):
        if column_name is not self.cfg.category_column_name:
            return torch.cat([item[column_name] for item in batch], dim=0)
        else:
            return [item[column_name] for item in batch]

    def collate_fn(self, batch):
        input_ids = self.simple_collate(batch, self.cfg.input_ids_column_name)
        pos_input_ids = self.simple_collate(batch, self.cfg.pos_input_ids_column_name)
        neg_input_ids = self.simple_collate(batch, self.cfg.neg_input_ids_column_name)
        
        pixel_pos_values = self.simple_collate(batch, self.cfg.pixels_pos_column_name)
        pixel_neg_values = self.simple_collate(batch, self.cfg.pixels_neg_column_name)

        pixel_pos_values = pixel_pos_values.to(memory_format=torch.contiguous_format).float()
        pixel_neg_values = pixel_neg_values.to(memory_format=torch.contiguous_format).float()
        
        collated = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pos_input_ids_column_name: pos_input_ids,
            self.cfg.neg_input_ids_column_name: neg_input_ids,
            self.cfg.pixels_pos_column_name: pixel_pos_values,
            self.cfg.pixels_neg_column_name: pixel_neg_values,
        }
        
        if self.split is not self.cfg.train_split_name:
            category = self.simple_collate(batch, self.cfg.category_column_name)

            collated = {
                self.cfg.input_ids_column_name: input_ids,
                self.cfg.pos_input_ids_column_name: pos_input_ids,
                self.cfg.neg_input_ids_column_name: neg_input_ids,
                self.cfg.pixels_pos_column_name: pixel_pos_values,
                self.cfg.pixels_neg_column_name: pixel_neg_values,
                self.cfg.category_column_name: category
            }
        return collated

    def __len__(self):
        return len(self.dataset)
