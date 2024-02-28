import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.config import ViTSAERunnerConfig
from tqdm import tqdm, trange


class ViTActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs. 
    """
    def __init__(
        self, cfg: ViTSAERunnerConfig, model: HookedVisionTransformer, create_dataloader: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((self.cfg.image_width, self.cfg.image_height)),  # Resize the image to WxH pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        self.iterable_dataset = iter(self.dataset)

        assert self.cfg.image_key in next(self.iterable_dataset).keys(), f'The image key \'{self.cfg.image_key}\' is not valid for this dataset.'
        
        if self.cfg.use_cached_activations:
            """
            Need to implement this. loads stored activations from a file.
            """
            pass
        
        if create_dataloader:
            if self.cfg.class_token:
              print("Starting to create the data loader!!!")
              self.dataloader = self.get_data_loader()
              print("Data loader created!!!")
            else:
              """
              Need to implement a buffer for the image patch training.
              """
              pass

    def get_image_batches(self):
        """
        Streams a batch of tokens from a dataset. returns a generator for efficient memory usage.
        """

        batch_size = self.cfg.store_size
        device = self.cfg.device
        images=[]
        for batch in trange(batch_size):
            next_image = next(self.iterable_dataset)[self.cfg.image_key]
            next_image = self.transform(next_image) # next_image is a torch tensor with size [C, W, H].
            images.append(next_image)
        batches = torch.stack(images, dim = 0) # batches has size [batch_size, C, W, H].
        batches = batches.to(device)
        return batches

    def get_activations(self, image_batches):
        
        module_name = self.cfg.module_name
        block_layer = self.cfg.block_layer
        list_of_hook_locations = [(block_layer, module_name)]

        activations = self.model.run_with_cache(
            image_batches,
            list_of_hook_locations,
        )[1][(block_layer, module_name)]
        
        if self.cfg.class_token:
          # Only keep the class token
          activations = activations[:,0,:] # See the forward(), foward_head() methods of the VisionTransformer class in timm. 
          # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension

        return activations
    
    def get_sae_batches(self):
        image_batches = self.get_image_batches()
        max_batch_size = self.cfg.max_batch_size_for_vit_forward_pass
        number_of_mini_batches = image_batches.size()[0] // max_batch_size
        remainder = image_batches.size()[0] % max_batch_size
        sae_batches = []
        for mini_batch in trange(number_of_mini_batches):
            sae_batches.append(self.get_activations(image_batches[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]))
        
        if remainder>0:
            sae_batches.append(self.get_activations(image_batches[-remainder:]))
            
        sae_batches = torch.cat(sae_batches, dim = 0)
        sae_batches = sae_batches.to(self.cfg.device)
        return sae_batches
        

    def get_data_loader(self) -> DataLoader:
        '''
        Return a torch.utils.dataloader which you can get batches from.
        
        Should automatically refill the buffer when it gets to n % full. 
        (better mixing if you refill and shuffle regularly).
        
        '''
        batch_size = self.cfg.batch_size
        
        sae_batches = self.get_sae_batches()
        
        dataloader = iter(DataLoader(sae_batches, batch_size=batch_size, shuffle=True))
        
        return dataloader
    
    
    def next_batch(self):
        """
        Get the next batch from the current DataLoader. 
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)