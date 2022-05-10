import torch
import torch.nn as nn
import numpy as np

from typing import Iterable, Tuple, Optional, Callable
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.transforms as transforms

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST


# MNIST conversion to RGB
def toRGBtorch(x):
    # C x H x W
    return x.repeat_interleave(3, 0)


# ================================================================================================
# regular distractor loaders (for Lewis signaling game, no caching or agent separation)
# ================================================================================
class DistractorsInputIterator(object):
    def __init__(self, data_dir, batch_size, num_distractors, device_id, num_gpus, seed=9):
        self.images_dir = data_dir
        self.batch_size = batch_size
        self.distractors = num_distractors
        self.imageset = ImageFolder(data_dir)
        self.data_set_len = len(self.imageset.samples)
        # self.samples = self.imageset.samples[self.data_set_len * device_id // num_gpus:
        #                                      self.data_set_len * (device_id + 1) // num_gpus]
        self.samples = self.imageset.samples
        self.n = len(self.samples)
        self.random_state = np.random.RandomState(seed)
        self.n_batches_per_epoch = np.ceil(len(self.samples) / self.batch_size).astype(int)
        self.idx = 0
        self.batches_generated = 0
        
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        
        if (self.idx >= len(self.samples) or self.batches_generated >= self.n_batches_per_epoch):
            self.batches_generated = 0
            self.idx = 0
            self.__iter__()
            raise StopIteration
        
        idxs = self.get_idxs()
        # t_idx = self.random_state.choice(self.distractors+1, size=self.batch_size)
        batch = []
        labels = []
        # chunk distractor batches manually without knowing read jpeg size
        for i, index in enumerate(idxs):
            jpeg_filename, label = self.samples[index]
            # variable length 
            bytes_i = np.fromfile(jpeg_filename, dtype=np.uint8)
            label_i = int(label)
            batch.append(bytes_i)
            labels.append(label_i)
        
        self.idx += self.batch_size  # only care about target ims
        self.batches_generated += 1
        
        labels = torch.from_numpy(np.asarray(labels))
        # print('labels')
        # print(labels)
        # print(torch.arange(len(labels), dtype=torch.uint8))
        return batch, labels  # torch.arange(len(labels), dtype=torch.uint8)
    
    def __len__(self):
        return self.data_set_len
    
    def get_idxs(self):
        if self.batch_size * (self.distractors+1) > self.n:
            print(f"WARN: {self.batch_size * (self.distractors+1)} > {self.n}")
        idxs = self.random_state.choice(range(self.n), size=(self.batch_size * (self.distractors+1)), replace=False)
        # print(idxs)
        # print('imageset.samples')
        # print([self.imageset.samples[i][1] for i in idxs])
        return idxs

    
def ExternalSourcePipeline(batch_size, num_distractors, num_threads, img_size, device_id, external_data,
                           mean=None, std=None):
    if mean is not None:  # 1-d mean -> 3-d
        mean = torch.tensor(mean).reshape((1, 1, 3))
    if std is not None:
        std = torch.tensor(std).reshape(1, 1, 3)
        
    pipe = Pipeline(batch_size * (num_distractors + 1), num_threads, device_id)
    # t_idx = external_data.random_state.choice(num_distractors+1, size=batch_size)
    
    with pipe:
        # A single sample
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        output = fn.cast(images, dtype=types.UINT8)
        output = output / 255.
        if mean is not None and std is not None:
            output = fn.normalize(output, mean=mean, stddev=std)
        # output = fn.reshape(output, src_dims=[2, 0, 1])  # make CHW
            
        pipe.set_outputs(output, labels)
    return pipe


class DistractorHybridLoader(DALIClassificationIterator):
    def __init__(self, batch_size, num_distractors, img_size, 
                 input_iterator, pipe, last_batch_padded, last_batch_policy):
        super(DistractorHybridLoader, self).__init__(pipe, last_batch_padded=False, 
                                                     last_batch_policy=LastBatchPolicy.PARTIAL)
        self.batch_size = batch_size
        self.num_distractors = num_distractors
        self.img_size = img_size
        self.ii = input_iterator
        self.dataset_size = len(self.ii.imageset.samples)
        self.n_batches_per_epoch = np.ceil(self.dataset_size / self.batch_size).astype(int)
        
    def __next__(self):
        """
        Post-process the Iterator result
        """
        data = super(DistractorHybridLoader, self).__next__()
        labels = data[0]['label']
        output = data[0]['data']
        # plot_single(output[0])
        bs = self.batch_size
        nd = self.num_distractors
        # C must be last here or else get wrong images from reshape
        data_shape = (self.img_size, self.img_size, 3)
        t_idx = self.ii.random_state.choice(nd+1, size=bs)
        sender_labels = labels.reshape((bs, nd+1))[torch.arange(bs), t_idx]
        receiver_input = output.reshape((bs, nd+1, *data_shape))
        target = receiver_input[torch.arange(bs), t_idx]
        
        # casts
        t_idx = torch.from_numpy(t_idx).long()
        sender_labels = sender_labels.long()
        # change to CHW for model
        receiver_input = receiver_input.permute(0, 1, 4, 2, 3).float()
        target = target.permute(0, 3, 1, 2).float()
        return (target, None), t_idx, (receiver_input, None), sender_labels
    
    def reset(self):
        super(DistractorHybridLoader, self).reset()

        

def get_distractor_loader(data_dir, batch_size, num_distractors, img_size, mean=None, std=None, seed=9):
    dii = DistractorsInputIterator(data_dir, batch_size, 
                                   num_distractors, 0, 1, seed=seed)
    pipe = ExternalSourcePipeline(batch_size=batch_size, 
                                  num_distractors=num_distractors, 
                                  num_threads=2, img_size=img_size, 
                                  device_id=0, external_data=dii,
                                  mean=mean, std=std)
    dhl = DistractorHybridLoader(batch_size, num_distractors, img_size, 
                                 dii, pipe,
                                 last_batch_padded=False, last_batch_policy=LastBatchPolicy.PARTIAL)

    return dhl


def distractor_train_loader(state, img_size, mean, std):
    if state['dataset'].lower() == 'mnist':
        train_dataset = MNIST(root='./data/mnist', 
                              train=True, 
                              transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                              download=True)
        return SeparatedDistractorFromMemoryLoader(train_dataset, state['train_batch_size'],
                                                   state['distractors'], img_size, 
                                                   None, None, 
                                                   state['seed'], state['device'])
    else:
        return get_distractor_loader(state['train_dir'], 
                                     state['train_batch_size'], 
                                     state['distractors'],
                                     img_size, 
                                     mean=mean, std=std, seed=state['seed'])


def distractor_test_loader(state, img_size, mean, std):
    if state['dataset'].lower() == 'mnist':
        test_dataset = MNIST(root='./data/mnist', 
                              train=False, 
                              transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                              download=True)
        return SeparatedDistractorFromMemoryLoader(test_dataset, state['test_batch_size'],
                                                   state['distractors'], img_size, 
                                                   None, None, 
                                                   state['seed'], state['device'])
    else:
        return get_distractor_loader(state['test_dir'], 
                                     state['test_batch_size'], 
                                     state['distractors'],
                                     img_size, 
                                     mean=mean, std=std, seed=state['seed'])


class InMemoryWrapper(torch.utils.data.DataLoader):
    # Dummy class for MNIST to act like DALI loaders
    def __init__(self, n_batches_per_epoch, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_batches_per_epoch = n_batches_per_epoch
        self.curr_epoch_mode = 'static'
        self.last_epoch_mode = 'static'
        self.dataset_size = len(self.dataset)

    def reset(self):
        return

    def start_epoch(self, mode):
        assert mode in ['semiotic', 'static', 'reification']
        self.curr_epoch_mode = mode

    def end_epoch(self, mode):
        assert mode in ['semiotic', 'static', 'reification']
        self.last_epoch_mode = mode


# ================================================================================================
# push loaders (classification task)
# ================================================================================
class InputIterator(object):
    def __init__(self, data_dir, batch_size, device_id, num_gpus, seed=9, shuffle=True):
        self.images_dir = data_dir
        self.batch_size = batch_size
        self.imageset = ImageFolder(data_dir)
        self.data_set_len = len(self.imageset.samples)
        # self.samples = self.imageset.samples[self.data_set_len * device_id // num_gpus:
        #                                      self.data_set_len * (device_id + 1) // num_gpus]
        self.samples = self.imageset.samples
        self.n = len(self.samples)
        self.random_state = np.random.RandomState(seed)
        self.n_batches_per_epoch = np.ceil(len(self.samples) / self.batch_size).astype(int)
        self.idx = 0
        self.batches_generated = 0
        self.shuffle = shuffle

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            self.random_state.shuffle(self.samples)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename, label = self.samples[self.i % self.n]
            batch.append(np.fromfile(jpeg_filename, dtype = np.uint8))  # we can use numpy
            labels.append(torch.tensor([int(label)], dtype = torch.uint8)) # or PyTorch's native tensors
            self.i += 1
        return batch, labels

    def __len__(self):
        return self.data_set_len

    
def PushPipeline(batch_size, num_threads, img_size, device_id, external_data):        
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        # A single sample, not normalized
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        output = fn.cast(images, dtype=types.UINT8)
        output = output / 255.
        # output = fn.reshape(output, src_dims=[2, 0, 1])  # make CHW
        
        pipe.set_outputs(output, labels)
    return pipe


class HybridLoader(DALIClassificationIterator):
    def __init__(self, batch_size, img_size, 
                 input_iterator, pipe, last_batch_padded, last_batch_policy, device):
        super(HybridLoader, self).__init__(pipe, last_batch_padded=last_batch_padded, 
                                               last_batch_policy=LastBatchPolicy.PARTIAL)
        self.batch_size = batch_size
        self.img_size = img_size
        self.ii = input_iterator
        self.dataset_size = len(self.ii.imageset.samples)
        self.n_batches_per_epoch = np.ceil(self.dataset_size / self.batch_size).astype(int)
        self.device = device
        self.last_batch_policy = last_batch_policy
        self.last_batch_padded = last_batch_padded
        
    def __next__(self):
        """
        Post-process the Iterator result
        """
        data = super(HybridLoader, self).__next__()
        labels = data[0]['label']
        output = data[0]['data']
        output = output.permute(0, 3, 1, 2).float()  # make CHW
        labels = labels.reshape((-1, )).long()
        # push code wants on cpu
        if self.device == 'cpu':
            return output.cpu(), labels.cpu()
        else:
            return output, labels


def get_push_loader(data_dir, batch_size, img_size, seed):
    assert 'augmented' not in data_dir, f"The augmented data set should not be used for push operation!"
    pii = InputIterator(data_dir, batch_size, 0, 1, seed=seed, shuffle=False)
    pipe = PushPipeline(batch_size=batch_size, 
                                  num_threads=2, img_size=img_size, 
                                  device_id=0, external_data=pii)
    phl = HybridLoader(batch_size, img_size, pii, pipe,
                       last_batch_padded=False, 
                       last_batch_policy=LastBatchPolicy.PARTIAL,
                       device='cpu')

    return phl


def push_train_loader(state, img_size, seed):
    if state['dataset'].lower() == 'mnist':
        train_push_dataset = MNIST(root='./data/mnist', 
                               train=True, 
                               transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                               download=True)
        n_batches_per_epoch = np.ceil(len(train_push_dataset) / state['train_batch_size']).astype(int)
        train_push_loader = InMemoryWrapper(
            n_batches_per_epoch,
            dataset=train_push_dataset, batch_size=state['train_batch_size'], 
            shuffle=False,
            num_workers=4, pin_memory=False)
        return train_push_loader
    else:
        return get_push_loader(state['train_push_dir'], 
                               state['train_batch_size'], 
                               img_size, seed=seed)


def push_test_loader(state, img_size, seed):
    if state['dataset'].lower() == 'mnist':
        test_push_dataset = MNIST(root='./data/mnist', 
                               train=False, 
                               transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                               download=True)
        n_batches_per_epoch = np.ceil(len(test_push_dataset) / state['test_batch_size']).astype(int)
        test_push_loader = InMemoryWrapper(
            n_batches_per_epoch,
            dataset=test_push_dataset, batch_size=state['test_batch_size'], 
            shuffle=False,
            num_workers=4, pin_memory=False)
        return test_push_loader
    else:
        return get_push_loader(state['test_push_dir'], 
                               state['test_batch_size'], 
                               img_size, seed=seed)


# unnormalized loader on gpu for statistic calculation
def _get_statistics_loader(data_dir, batch_size, img_size, seed):
    pii = InputIterator(data_dir, batch_size, 0, 1, seed=seed, shuffle=False)
    pipe = PushPipeline(batch_size=batch_size, 
                                  num_threads=2, img_size=img_size, 
                                  device_id=0, external_data=pii)
    phl = HybridLoader(batch_size, img_size, pii, pipe,
                       last_batch_padded=False, 
                       last_batch_policy=LastBatchPolicy.PARTIAL,
                       device='gpu')

    return phl


def statistics_train_loader(state, img_size, seed):
    return _get_statistics_loader(state['train_dir'], 
                                  state['train_batch_size'], 
                                  img_size, seed=seed)



# ================================================================================================
# normalized loaders  (classification task)
# ================================================================================
def NormalizedPipeline(batch_size, num_threads, img_size, device_id, external_data, mean, std):
    if mean is not None:  # 1-d mean -> 3-d
        mean = torch.tensor(mean).reshape((1, 1, 3))
    if std is not None:
        std = torch.tensor(std).reshape((1, 1, 3))
        
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        # A single sample, not normalized
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        output = fn.cast(images, dtype=types.UINT8)
        output = output / 255.
        # output = fn.reshape(output, src_dims=[2, 0, 1])  # make CHW
        if mean is not None and std is not None:
            output = fn.normalize(output, mean=mean, stddev=std)
            
        pipe.set_outputs(output, labels)
    return pipe


def get_normalized_loader(data_dir, batch_size, img_size, seed, mean, std, shuffle=True):
    pii = InputIterator(data_dir, batch_size, 0, 1, seed=seed, shuffle=shuffle)
    pipe = NormalizedPipeline(batch_size=batch_size, 
                              num_threads=2, img_size=img_size, 
                              device_id=0, external_data=pii, mean=mean, std=std)
    phl = HybridLoader(batch_size, img_size, pii, pipe,
                       last_batch_padded=False, 
                       last_batch_policy=LastBatchPolicy.PARTIAL,
                       device='mixed')

    return phl


def normalized_train_loader(state, img_size, seed, mean, std, shuffle=True):
    if state['dataset'].lower() == 'mnist':
        # in-memory loader 
        train_dataset = MNIST(root='./data/mnist', 
                            train=True, 
                            transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                            download=True)
        n_batches_per_epoch = np.ceil(len(train_dataset) / state['train_batch_size']).astype(int)
        train_loader = InMemoryWrapper(
                            n_batches_per_epoch,
                            dataset=train_dataset, batch_size=state['train_batch_size'], 
                            shuffle=shuffle,
                            num_workers=4, pin_memory=False)
        return train_loader
    else:
        return get_normalized_loader(state['train_dir'], 
                                     state['train_batch_size'], 
                                     img_size, seed=seed, mean=mean, std=std, shuffle=shuffle)


def normalized_test_loader(state, img_size, seed, mean, std):
    if state['dataset'].lower() == 'mnist':
        test_dataset = MNIST(root='./data/mnist', 
                            train=False, 
                            transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]),
                            download=True)
        n_batches_per_epoch = np.ceil(len(test_dataset) / state['test_batch_size']).astype(int)
        test_loader = InMemoryWrapper(
                            n_batches_per_epoch,
                            dataset=test_dataset, batch_size=state['test_batch_size'], 
                            shuffle=False,
                            num_workers=4, pin_memory=False)
        return test_loader
    else:
        return get_normalized_loader(state['test_dir'], 
                                    state['test_batch_size'], 
                                    img_size, seed=seed, mean=mean, std=std, shuffle=False)



# ================================================================================================
# cached hybrid loader (signaling game)
# ================================================================================
class DistractorHybridCachedLoader(DALIGenericIterator):
    def __init__(self, batch_size, num_distractors, img_size, 
                 input_iterator, pipe, last_batch_padded, last_batch_policy, preprocess, x_loader,
                 iterator_keywords=["data", "label"]):
        """
        preprocess: Callable for features
        x_loader: data loader for caching features
        """
        super(DistractorHybridCachedLoader, self).__init__(pipe, iterator_keywords,
                                                           last_batch_padded=False, 
                                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                                           prepare_first_batch=False)
        self.batch_size = batch_size
        self.num_distractors = num_distractors
        self.img_size = img_size
        self.ii = input_iterator
        self.dataset_size = len(self.ii.imageset.samples)
        self.n_batches_per_epoch = np.ceil(self.dataset_size / self.batch_size).astype(int)
        self.last_epoch_mode = 'static'
        self.curr_epoch_mode = None
        self.x_loader = x_loader
        assert x_loader.ii.shuffle == False, "Make sure cache loader is not shuffling!"
        assert x_loader.last_batch_policy == LastBatchPolicy.PARTIAL
        assert x_loader.last_batch_padded == False
        self.preprocess = preprocess
        self.cached_batches_generated = 0
        self.cached_idx = 0
        if preprocess:
            self.cache = self.generate_cache(self.preprocess)
            self.use_cache = True
        else:
            self.cache = None
            self.use_cache = False
        
    def __next__(self):
        """
        get from cache or re-cache based on simple FSM
        """
        if self.last_epoch_mode == None:
            raise RuntimeError("Update mode with start_epoch before getting data.")

        bs = self.batch_size
        nd = self.num_distractors
        
        if self.use_cache:  # updated in self.start_epoch(...)
            # print("Using cache")
            self.check_stop_condition()
            labels = self.cache['labels']
            output = self.cache['feats']
            structures = self.cache['structures']
            
            idxs = self.ii.get_idxs()  # re-use input iterator's seed
            self.cached_batches_generated += 1
            self.cached_idx += bs  # only count target ims
            # B*K, V
            output = output[idxs].cuda()
            labels = labels[idxs].cuda()
            structures = structures[idxs].cuda()
            
            # print("labels use_cache=True")
            # print(labels.reshape((-1,)))
            data_shape = output.shape[1:]
            struct_shape = structures.shape[1:]
            receiver_structs = structures.reshape((bs, nd+1, *struct_shape))
            
        else:
            # print("Not using cache")
            data = super(DistractorHybridCachedLoader, self).__next__()
            labels = data[0]['label']
            output = data[0]['data']
            structures = None # not used if nonstatic epoch 
            
            # Note: input iterator runs "ahead" of the actual iteration
            # print("labels use_cache=False")
            # print(labels)

            # C must be last here or else get wrong images from reshape
            data_shape = (self.img_size, self.img_size, 3)
            receiver_structs = None
            t_structs = None
        
        # lewis signaling game target indices
        t_idx_lewis_game = self.ii.random_state.choice(nd+1, size=bs)
        # concept labels for diagnostic classifier training 
        sender_labels = labels.reshape((bs, nd+1))[torch.arange(bs), t_idx_lewis_game]

        receiver_input = output.reshape((bs, nd+1, *data_shape))
        t_objects = receiver_input[torch.arange(bs), t_idx_lewis_game]
        
        if receiver_structs is not None:
            t_structs = receiver_structs[torch.arange(bs), t_idx_lewis_game]

        # casts
        t_idx_lewis_game = torch.from_numpy(t_idx_lewis_game).long()
        sender_labels = sender_labels.long()
        receiver_input = receiver_input.float()
        t_objects = t_objects.float()
        
        if not self.use_cache:
            # change to CHW for model
            receiver_input = receiver_input.permute(0, 1, 4, 2, 3)
            t_objects = t_objects.permute(0, 3, 1, 2)
        else:
            receiver_structs = receiver_structs.float()
            t_structs = t_structs.float()

        return (t_objects, t_structs), t_idx_lewis_game, (receiver_input, receiver_structs), sender_labels
            
    def start_epoch(self, mode):
        assert mode in ['semiotic', 'static',  'reification']
        self.curr_epoch_mode = mode
        
        self.use_cache = False
        if self.last_epoch_mode == 'semiotic' and self.curr_epoch_mode == 'static':
            # regenerate cache with updated signs model
            self.cache = self.generate_cache(self.preprocess)
            self.use_cache = True
        if self.last_epoch_mode == 'static' and self.curr_epoch_mode == 'static':
            # No change in signs model, don't re-cache
            self.use_cache = True
        if self.curr_epoch_mode == 'semiotic':
            self.use_cache = False
        
    def end_epoch(self, mode):
        assert mode in ['semiotic', 'static']
        self.last_epoch_mode = mode
        
    def generate_cache(self, preprocess, log=print):
        log(f"Updating cache for non-semiotic epochs...")
        cache = None
        loader = self.x_loader
        with tqdm(total=self.dataset_size) as pb:
            start = 0
            for i, data in enumerate(loader):
                outputs, labels = data
                end = min(start + len(outputs), self.dataset_size)
                with torch.no_grad():
                    feats, structures = preprocess(outputs)
                
                if cache is None:
                    cache = {
                        'feats': torch.zeros((self.dataset_size, *feats.shape[1:])),
                        'structures': torch.zeros((self.dataset_size, *structures.shape[1:])),
                        'labels': torch.zeros(self.dataset_size).long()
                    }  
                cache['feats'][start:end] = feats.detach().cpu()[:end-start]  # stop at padded samples (last iter)
                cache['structures'][start:end] = structures.detach().cpu()[:end-start]  # stop at padded samples (last iter)
                cache['labels'][start:end] = labels.cpu()[:end-start]
                pb.update(end-start)
                
                start = end
        
            
        self.x_loader.reset()
        return cache
    
    def check_stop_condition(self):
        """
        Manual stop when using cache, otherwise handled by self.input_iterator
        """
        if (self.cached_idx >= self.dataset_size or self.cached_batches_generated >= self.n_batches_per_epoch):
            self.cached_batches_generated = 0
            self.cached_idx = 0
            self.__iter__()
            raise StopIteration
    

def get_cached_distractor_loader(data_dir, batch_size, num_distractors, img_size, preprocess,
                                 mean=None, std=None, seed=9):
    dii = DistractorsInputIterator(data_dir, batch_size, 
                                   num_distractors, 0, 1, seed=seed)
    pipe = ExternalSourcePipeline(batch_size=batch_size, 
                                  num_distractors=num_distractors, 
                                  num_threads=2, img_size=img_size, 
                                  device_id=0, external_data=dii,
                                  mean=mean, std=std)
    x_loader = get_normalized_loader(data_dir, batch_size, img_size, seed, mean, std, shuffle=False)
    dhl = DistractorHybridCachedLoader(batch_size, num_distractors, img_size, 
                                       dii, pipe,
                                       last_batch_padded=False, 
                                       last_batch_policy=LastBatchPolicy.PARTIAL,
                                       preprocess=preprocess, 
                                       x_loader=x_loader)

    return dhl


def cached_distractor_train_loader(state, preprocess, img_size, mean, std):
    return get_cached_distractor_loader(state['train_dir'], 
                                        state['train_batch_size'], 
                                        state['distractors'],
                                        img_size, 
                                        mean=mean, std=std, seed=state['seed'], preprocess=preprocess)


def cached_distractor_test_loader(state, preprocess, img_size, mean, std):
    return get_cached_distractor_loader(state['test_dir'], 
                                        state['test_batch_size'], 
                                        state['distractors'],
                                        img_size, 
                                        mean=mean, std=std, seed=state['seed'], preprocess=preprocess)


# ================================================================================================
# logically separated cached hybrid loader (sender and receiver have separate percept models)
# ================================================================================
class SeparatedDistractorsInputIterator(DistractorsInputIterator):
    def __next__(self):
        
        if (self.idx >= len(self.samples) or self.batches_generated >= self.n_batches_per_epoch):
            self.batches_generated = 0
            self.idx = 0
            self.__iter__()
            raise StopIteration
        
        bs = self.batch_size
        nd = self.distractors
        idxs = self.get_idxs()
        idxs = idxs.reshape((bs, nd+1))  # pipeline expects every output to be batch_size long
        t_idx_lewis_game = self.random_state.choice(nd+1, size=bs)  # game indices
        t_idx_concepts = idxs[torch.arange(bs), t_idx_lewis_game]  # dataset indices
        batch = []
        labels = []
        # chunk distractor batches manually without knowing read jpeg size
        for i, index in enumerate(t_idx_concepts.flatten()):
            jpeg_filename, label = self.samples[index]
            # variable length 
            bytes_i = np.fromfile(jpeg_filename, dtype=np.uint8)
            label_i = int(label)
            batch.append(bytes_i)
            labels.append(label_i)
        
        self.idx += self.batch_size  # only care about target ims
        self.batches_generated += 1
        
        labels = torch.from_numpy(np.asarray(labels))
        # print('labels')
        # print(labels)
        # print(torch.arange(len(labels), dtype=torch.uint8))
        return batch, labels, idxs, t_idx_lewis_game  # torch.arange(len(labels), dtype=torch.uint8)

    
def SeparatedExternalSourcePipeline(batch_size, num_distractors, num_threads, img_size, device_id, external_data,
                                    mean=None, std=None):
    if mean is not None:  # 1-d mean -> 3-d
        mean = torch.tensor(mean).reshape((1, 1, 3))
    if std is not None:
        std = torch.tensor(std).reshape(1, 1, 3)
    
    # We only need batch size (t_b) per input stream
    pipe = Pipeline(batch_size, num_threads, device_id)
    
    with pipe:
        # A single sample
        jpegs, labels, idxs, t_idx = fn.external_source(source=external_data, num_outputs=4)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        output = fn.cast(images, dtype=types.UINT8)
        output = output / 255.
        if mean is not None and std is not None:
            output = fn.normalize(output, mean=mean, stddev=std)
        # output = fn.reshape(output, src_dims=[2, 0, 1])  # make CHW
            
        pipe.set_outputs(output, labels, idxs, t_idx)
    return pipe


class SeparatedDistractorHybridCachedLoader(DistractorHybridCachedLoader):
    def __init__(self, batch_size, num_distractors, img_size, 
                 input_iterator, pipe, last_batch_padded, last_batch_policy, 
                 sender_preprocess, receiver_preprocess, x_loader, log=print):
        """
        sender_preprocess: Callable for sender features
        receiver_preprocess: Callable for receiver features
        x_loader: data loader for caching features
        """
        super(SeparatedDistractorHybridCachedLoader, self).__init__(
                batch_size, num_distractors, img_size, input_iterator, pipe, 
                last_batch_padded, last_batch_policy, None, x_loader, 
                iterator_keywords=["data", "label", "all_idxs", "target_idx"])
        self.log = log
        self.sender_preprocess = sender_preprocess
        self.receiver_preprocess = receiver_preprocess
        
        log("Initiating the sender cache...")
        self.sender_cache = self.generate_cache(self.sender_preprocess)
        log("Initiating the receiver cache once.")
        self.receiver_cache = self.generate_cache(self.receiver_preprocess)
        # Default behavior is to use cache
        self.use_cache = True
        
    def __next__(self):
        """
        get from cache or re-cache based on simple FSM
        (t_objects, t_structs): tuple of objects (feats or images, depending on FSM), and their disentangled structure
        t_idx: the indices or "labels" which index the described object within distractor set
        (receiver_input, receiver_structs):  tuple of objects (feats or images)  and disentangled structure
        sender_labels: object labels from the upstream classification task (only used for diagnostic classifier)
        
        """
        if self.last_epoch_mode == None:
            raise RuntimeError("Update mode with start_epoch before getting data.")

        bs = self.batch_size
        nd = self.num_distractors

        receiver_feats_shape = self.receiver_cache['feats'].shape[1:]
        
        if self.use_cache:  # updated in self.start_epoch(...)
            self.check_stop_condition()
            labels = self.sender_cache['labels']
            output = self.sender_cache['feats']
            data_shape = output.shape[1:]
            
            idxs = self.ii.get_idxs()  # re-use input iterator's seed
            # The target indices in our Lewis signaling game batch
            t_idx_lewis_game = self.ii.random_state.choice(nd+1, size=bs)
            # index for current set to grab sender's objects and upstream concept labels 
            sender_idxs = idxs.reshape((bs, nd+1))[torch.arange(bs), t_idx_lewis_game]
            sender_idxs = sender_idxs.flatten()
            
            self.cached_batches_generated += 1
            self.cached_idx += bs  # only count target ims
            # B, V
            t_objects = output[sender_idxs].cuda()
            # B, 1
            # The indices from original concept dataset (for diagnostic classifier training)
            sender_labels = labels[sender_idxs]
            
            # handle disentangled structures from the sender percept models
            t_structs = self.sender_cache['structures'][sender_idxs].cuda()
            
            t_idx_lewis_game = torch.from_numpy(t_idx_lewis_game)
        else:
            # Note: input iterator runs "ahead" of the actual iteration
            data = DALIGenericIterator.__next__(self)
            sender_labels = data[0]['label'] 
            t_objects = data[0]['data']
            idxs = data[0]['all_idxs'].cpu().reshape((bs * (nd + 1)))
            t_idx_lewis_game = data[0]['target_idx'].cpu()
            
            # C must be last here or else get wrong images from reshape
            sender_shape = (self.img_size, self.img_size, 3)
            t_structs = None
            receiver_structs = None
            
        # always use the cached versions for receiver
        receiver_input = self.receiver_cache['feats'][idxs].reshape((bs, nd+1, *receiver_feats_shape))
        receiver_structs = self.receiver_cache['structures'][idxs].reshape((bs, nd+1, -1)).cuda()

        # casts
        sender_labels = sender_labels.long()
        receiver_input = receiver_input.float()
        receiver_structs = receiver_structs.float()
        t_objects = t_objects.float()
        t_idx_lewis_game = t_idx_lewis_game.long()
        
        if not self.use_cache:
            # change to CHW for model
            t_objects = t_objects.permute(0, 3, 1, 2)
        else:
            t_structs = t_structs.float()
        
        return (t_objects, t_structs), t_idx_lewis_game, (receiver_input, receiver_structs), sender_labels
            
    def start_epoch(self, mode):
        assert mode in ['semiotic', 'static']
        self.curr_epoch_mode = mode
        
        self.use_cache = False
        
        if self.last_epoch_mode == 'semiotic' and self.curr_epoch_mode == 'static':
            # regenerate cache with updated signs model
            self.sender_cache = self.generate_cache(self.sender_preprocess)
            # You would regenerate receiver cache as well if you needed it.
            self.use_cache = True
        if self.last_epoch_mode == 'static' and self.curr_epoch_mode == 'static':
            # No change in signs model, don't re-cache
            self.use_cache = True
        if self.curr_epoch_mode == 'semiotic':
            self.use_cache = False

    
def get_separated_cached_distractor_loader(data_dir, batch_size, num_distractors, img_size, 
                                           sender_preprocess, receiver_preprocess,
                                           mean=None, std=None, seed=9):
    dii = SeparatedDistractorsInputIterator(data_dir, batch_size, 
                                            num_distractors, 0, 1, seed=seed)
    pipe = SeparatedExternalSourcePipeline(batch_size=batch_size, 
                                           num_distractors=num_distractors, 
                                           num_threads=2, img_size=img_size, 
                                           device_id=0, external_data=dii,
                                           mean=mean, std=std)
    x_loader = get_normalized_loader(data_dir, batch_size, img_size, seed, mean, std, shuffle=False)
    dhl = SeparatedDistractorHybridCachedLoader(batch_size, num_distractors, img_size, 
                                                dii, pipe,
                                                last_batch_padded=False, 
                                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                                sender_preprocess=sender_preprocess, 
                                                receiver_preprocess=receiver_preprocess,
                                                x_loader=x_loader)

    return dhl


### MNIST 
class SeparatedDistractorFromMemoryLoader(object):
    def __init__(self, dataset, batch_size, num_distractors, img_size, 
                 sender_preprocess, receiver_preprocess, seed, device, log=print):
        self.dataset = dataset
        self.samples = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.n_batches_per_epoch = np.ceil(len(self.samples) / self.batch_size).astype(int)
        self.distractors = num_distractors
        self.img_size = img_size
        self.sender_preprocess = sender_preprocess
        self.receiver_preprocess = receiver_preprocess
        self.curr_epoch_mode = 'static'
        self.last_epoch_mode = 'static'
        self.log = log
        self.seed = seed
        self.device = device
        self.random_state = np.random.RandomState(seed)

        assert self.distractors * self.batch_size < len(self.dataset), \
        f"Batch size or distractors is too large for this population size: " \
        f"{self.distractors}*{self.batch_size} > {len(self.dataset)}."

    def __iter__(self):
        if self.last_epoch_mode == None:
            raise RuntimeError("Update mode with start_epoch before getting data.")
        self.seed += 1  # shuffle on starting a new epoch
        return MemoryIterator(
            self.dataset, self.batch_size, self.distractors, self.img_size, 
            self.curr_epoch_mode, self.sender_preprocess, self.receiver_preprocess, self.seed, self.device
        )

    def reset(self):
        return

    def start_epoch(self, mode):
        assert mode in ['semiotic', 'static', 'reification']
        self.curr_epoch_mode = mode

    def end_epoch(self, mode):
        assert mode in ['semiotic', 'static', 'reification']
        self.last_epoch_mode = mode


class MemoryIterator(object):
    def __init__(self, dataset, batch_size, num_distractors, img_size, curr_epoch_mode,
                 sender_preprocess, receiver_preprocess, seed, device):
        self.batches_generated = 0
        self.idx = 0
        self.dataset = dataset
        self.samples = dataset
        self.batch_size = batch_size
        self.n_batches_per_epoch = np.ceil(len(self.samples) / self.batch_size).astype(int)
        self.distractors = num_distractors
        self.curr_epoch_mode = curr_epoch_mode
        self.img_size = img_size
        self.sender_preprocess = sender_preprocess
        self.device = device
        self.receiver_preprocess = receiver_preprocess
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.idx >= len(self.dataset) or self.batches_generated >= self.n_batches_per_epoch):
            self.batches_generated = 0
            self.idx = 0
            raise StopIteration()
        
        idxs = self.random_state.choice(range(len(self.dataset)), size=(self.batch_size * (self.distractors+1)), replace=False)
        t_idx_lewis_game = self.random_state.choice(self.distractors+1, size=self.batch_size)

        samples = [self.dataset[i] for i in idxs]
        samples_ = torch.stack([sample[0] for sample in samples])
        raw_shape = samples_[0].shape
        raw_input = samples_.reshape((self.batch_size, self.distractors+1, *raw_shape))
        raw_input = raw_input.float()

        sender_input = raw_input[torch.arange(self.batch_size), t_idx_lewis_game]
        if self.sender_preprocess is not None and self.curr_epoch_mode == 'static':
            with torch.no_grad():
                t_objects, t_structs = self.sender_preprocess(sender_input.to(self.device))
        else:
            t_objects, t_structs = sender_input, None

        labels_ = torch.tensor([sample[1] for sample in samples])
        labels_ = labels_.reshape((self.batch_size, self.distractors+1))

        sender_labels = labels_[torch.arange(self.batch_size), t_idx_lewis_game]
        sender_labels = sender_labels.long()
        t_idx_lewis_game = torch.from_numpy(t_idx_lewis_game).long()

        if self.receiver_preprocess is not None and self.curr_epoch_mode == 'static':
            with torch.no_grad():
                receiver_input, receiver_structs = self.receiver_preprocess(raw_input.to(self.device))
        else:
            receiver_input, receiver_structs = raw_input, None        

        self.batches_generated += 1
        self.idx += self.batch_size

        return (t_objects, t_structs), t_idx_lewis_game, (receiver_input, receiver_structs), sender_labels
        

def separated_cached_distractor_train_loader(state, sender_preprocess, receiver_preprocess, 
                                             img_size, mean, std):
    if state['dataset'].lower() == 'mnist':
        train_dataset = MNIST(root='./data/mnist', 
                              train=True, 
                              transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                              download=True)
        return SeparatedDistractorFromMemoryLoader(train_dataset, state['train_batch_size'],
                                                   state['distractors'], img_size, 
                                                   sender_preprocess, receiver_preprocess, 
                                                   state['seed'], state['device'])
    else:
        return get_separated_cached_distractor_loader(state['train_dir'], 
                                                      state['train_batch_size'], 
                                                      state['distractors'],
                                                      img_size, 
                                                      mean=mean, std=std, seed=state['seed'],
                                                      sender_preprocess=sender_preprocess,
                                                      receiver_preprocess=receiver_preprocess)


def separated_cached_distractor_test_loader(state, sender_preprocess, receiver_preprocess,
                                            img_size, mean, std):
    if state['dataset'].lower() == 'mnist':
        test_dataset = MNIST(root='./data/mnist', 
                              train=False, 
                              transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                              download=True)
        return SeparatedDistractorFromMemoryLoader(test_dataset, state['test_batch_size'],
                                                   state['distractors'], img_size, 
                                                   sender_preprocess, receiver_preprocess, 
                                                   state['seed'], state['device'])
    else:
        return get_separated_cached_distractor_loader(state['test_dir'], 
                                                      state['test_batch_size'], 
                                                      state['distractors'],
                                                      img_size, 
                                                      mean=mean, std=std, seed=state['seed'],
                                                      sender_preprocess=sender_preprocess,
                                                      receiver_preprocess=receiver_preprocess)
