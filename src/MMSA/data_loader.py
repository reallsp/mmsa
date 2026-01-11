import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2':self.__init_simsv2,
            'custom': self.__init_custom,
            'train_12_16': self.__init_custom,
            'copa_1231': self.__init_custom,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        if self.args['custom_feature']:
            # use custom feature file extracted with MMSA-FET
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # use deault feature file specified in config file
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if 'sims' in self.args['dataset_name']:
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A']:
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V']:
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.get('data_missing'):
            # Currently only support unaligned data missing.
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args['need_data_aligned']:
                self.audio_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')

        if self.args.get('need_normalized'):
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_simsv2(self):
        return self.__init_mosi()
    
    def __init_custom(self):
        """初始化自定义数据集 (支持独立模式文件加载)"""
        feature_path = self.args.get('custom_feature') or self.args.get('featurePath')
        
        # 支持路径占位符，如 "copa_{mode}_converted.pkl"
        if "{mode}" in str(feature_path):
            current_file = str(feature_path).format(mode=self.mode)
        else:
            # 如果是独立文件模式，尝试查找针对当前 mode 的文件
            # 例如 featurePath 是 "copa_1231_converted.pkl"，而我们需要加载 "train"
            # 逻辑：如果直接加载失败且文件不存在，尝试寻找包含 mode 名的文件
            current_file = feature_path
            if not os.path.exists(current_file):
                # 尝试自动补全模式名，例如：/path/to/data_train.pkl
                base, ext = os.path.splitext(current_file)
                mode_specific = f"{base}_{self.mode}{ext}"
                if os.path.exists(mode_specific):
                    current_file = mode_specific
                elif self.mode == 'valid' and not os.path.exists(current_file):
                    # 如果验证集文件不存在，尝试用测试集代替
                    mode_specific = f"{base}_test{ext}"
                    if os.path.exists(mode_specific):
                        current_file = mode_specific

        logger.info(f"Loading {self.mode} data from {current_file}")
        with open(current_file, 'rb') as f:
            data = pickle.load(f)
        
        # 兼容性处理：有些 pkl 只有单层 dict，有些包裹了 mode 层
        if self.mode in data:
            data_mode = data[self.mode]
        elif len(data.keys()) == 1:
            # 如果只有一个 key（如 'test'），直接使用该 key 的内容
            data_mode = list(data.values())[0]
        else:
            data_mode = data
        
        # 加载文本特征
        # 优先使用text字段（序列格式 [N, seq_len, dim]），如果不支持BERT的模型使用
        # 如果use_bert为True，则使用text_bert字段（[N, 3, seq_len]）
        if self.args.get('use_bert', None) and 'text_bert' in data_mode:
            self.text = data_mode['text_bert'].astype(np.float32)
            # text_bert格式是[N, 3, seq_len]
            # 特征维度设置为768（BERT维度）
            self.args['feature_dims'][0] = 768
        elif 'text' in data_mode:
            self.text = data_mode['text'].astype(np.float32)
            # text格式是[N, seq_len, dim]
            self.args['feature_dims'][0] = self.text.shape[2] if len(self.text.shape) > 2 else self.text.shape[1]
        elif 'text_bert' in data_mode:
            # 如果没有text字段但有text_bert，作为fallback使用text_bert
            self.text = data_mode['text_bert'].astype(np.float32)
            # 特征维度设置为768（BERT维度）
            self.args['feature_dims'][0] = 768
        else:
            raise ValueError(f"No text feature found in {self.mode} data")
        
        # 加载音频特征
        self.audio = data_mode['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        
        # 加载视觉特征
        self.vision = data_mode['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        
        # 加载额外模态 (保留全部特征)
        self.extra_features = {}
        for k in ['ir_feature', 'bio', 'eye', 'eeg', 'eda']:
            if k in data_mode:
                self.extra_features[k] = data_mode[k].astype(np.float32)
                if not self.args['need_data_aligned']:
                    self.extra_features[f'{k}_lengths'] = data_mode.get(f'{k}_lengths', [self.extra_features[k].shape[1]] * len(self.audio))
        
        # 加载其他字段
        self.raw_text = data_mode.get('raw_text', [''] * len(self.audio))
        self.ids = data_mode.get('id', [f'sample_{i}' for i in range(len(self.audio))])
        
        # 加载标签
        self.labels = {
            'M': np.array(data_mode['regression_labels']).astype(np.float32)
        }
        
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
        
        # 加载序列长度信息（如果存在）
        if not self.args['need_data_aligned']:
            self.audio_lengths = data_mode.get('audio_lengths', [self.audio.shape[1]] * len(self.audio))
            self.vision_lengths = data_mode.get('vision_lengths', [self.vision.shape[1]] * len(self.vision))
            # 确保audio_lengths是列表
            if isinstance(self.audio_lengths, np.ndarray):
                self.audio_lengths = list(self.audio_lengths)
            if isinstance(self.vision_lengths, np.ndarray):
                self.vision_lengths = list(self.vision_lengths)
        
        # 处理-inf值
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0
        
        # 如果需要归一化（与 aligned 兼容，做时间维均值，输出形状 (N,1,dim)）
        if self.args.get('need_normalized'):
            self.__normalize()
        
        # 如果需要归一化
        if self.args.get('need_normalized'):
            self.__normalize()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        # NOTE: truncate input to specific length.
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        
        # 添加额外模态
        if hasattr(self, 'extra_features'):
            for k, v in self.extra_features.items():
                if k.endswith('_lengths'):
                    sample[k] = v[index]
                else:
                    sample[k] = torch.Tensor(v[index])

        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        if self.args.get('data_missing'):
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['audio_mask'] = self.audio_mask[index]
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['vision_mask'] = self.vision_mask[index]
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])

        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
