from torch.utils.data import Dataset

class RedisDataset(Dataset):
    def __init__(self,ranked_knobs_info,external_metric):
        super(RedisDataset,self).__init__()
        self.ranked_knobs_info = ranked_knobs_info
        self.external_metric = external_metric
        assert len(ranked_knobs_info) == len(external_metric)
    
    def __len__(self):
        return len(self.ranked_knobs_info)

    def __getitem__(self,idx):
        return self.ranked_knobs_info[idx],self.external_metric[idx]
