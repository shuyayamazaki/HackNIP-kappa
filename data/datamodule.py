from typing import Any, Dict, List
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers.data.data_collator import default_data_collator
from torch_geometric.data import Data, Batch


class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            debug: bool = False,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.data_path = {
            'train': data_path+'_train.parquet',
            'val': data_path+'_val.parquet',
            'test': data_path+'_test.parquet',
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)
        
        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset
 
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    

class MultiModalDataModule(BaseDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 12,
            tokenizer_model: str = 'bert-base-uncased',
            *args,
            **kwargs,
        ):
        super().__init__(data_path, batch_size, num_workers, tokenizer_model, *args, **kwargs)
        self.graph_data_collator = graph_data_collator
        self.text_data_collator = text_data_collator
        self.token_fn = lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=512)

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        graph_batch = self.graph_data_collator(features)
        text_batch = self.text_data_collator(features, self.token_fn)
        return graph_batch, text_batch
    

def graph_data_collator(features: List[dict]) -> Dict[str, Any]:
    """
    """
    return Batch.from_data_list([Data(x=torch.tensor(f["node_feat"], dtype=torch.float32), 
                                      edge_index=torch.tensor(f['edge_index']), 
                                      edge_attr=torch.tensor(f['edge_attr'], dtype=torch.float32),
                                      y=torch.tensor(f['y'], dtype=torch.float32)) for f in features])


def text_data_collator(features: List[dict], token_fn) -> Dict[str, Any]:
    '''
    '''
    text_batch = [token_fn(f['text']) for f in features]
    return default_data_collator(text_batch)


class GraphSupervisedDataModule(BaseDataModule):
    def __init__(
            self, 
            cfg: DictConfig,
            *args,
            **kwargs,
        ):
        self.cfg = cfg
        data_path = cfg.data_dir
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        label = cfg.label
        task = cfg.task
        super().__init__(data_path, batch_size, num_workers, *args, **kwargs)
        self.label = label
        self.task = task

    def setup(self, stage=None):
        dataset = load_dataset('parquet', data_files=self.data_path)
        if self.task == 'classification':
            from sklearn.preprocessing import LabelEncoder
            self.categories = dataset['train'].unique(self.label)
            self.label_encoder = LabelEncoder().fit(self.categories)
        
        dataset['train'] = dataset['train'].train_test_split(test_size=0.995)['train'] if self.debug else dataset['train']
        dataset['train'] = dataset['train'].shuffle()
        self.dataset = dataset

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        if self.task == 'classification':
            for f in features:
                f['y'] = self.label_encoder.transform([f[self.label]])[0]
        elif self.task == 'regression':
            for f in features:
                y = f[self.label]
                f['y'] = float(y)
        graph_batch = self.graph_data_collator(features)
        return graph_batch
    
    def graph_data_collator(self, features: List[dict]) -> Dict[str, Any]:
        """
        """
        return graph_data_collator(features)


class IonConductivityDataModule(GraphSupervisedDataModule):
    def __init__(
            self, 
            cfg: DictConfig,
            *args,
            **kwargs,
        ):
        super().__init__(cfg, *args, **kwargs)

    def collate_fn(self, features: List[dict]) -> Dict[str, Any]:
        if self.task == 'classification':
            for f in features:
                f['y'] = self.label_encoder.transform([f[self.label]])[0]
        elif self.task == 'regression':
            for f in features:
                y = f[self.label]
                f['y'] = float(y)
        graph_batch = self.graph_data_collator(features)
        return graph_batch
    

    def graph_data_collator(self, features: List[dict]) -> Dict[str, Any]:
        """
        """
        return Batch.from_data_list([Data(x=torch.tensor(f["node_feat"], dtype=torch.float32), 
                                        edge_index=torch.tensor(f['edge_index']), 
                                        edge_attr=torch.tensor(f['edge_attr'], dtype=torch.float32),
                                        temperature=torch.tensor(f['data_temperature_value'], dtype=torch.float32),
                                        y=torch.tensor(f['y'], dtype=torch.float32)) for f in features])
