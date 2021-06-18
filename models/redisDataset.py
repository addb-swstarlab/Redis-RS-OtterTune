from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class RedisDataset(Dataset):
    def __init__(self,ranked_knobs_info,external_metric):
        self.ranked_knobs_info = ranked_knobs_info
        self.external_metric = external_metric

        self.x_scaler = MinMaxScaler()
        self.x_scaler.fit(self.ranked_knobs_info)
        self.X_data = self.x_scaler.transform(self.ranked_knobs_info)

        self.y_scaler = MinMaxScaler()
        self.y_scaler.fit(self.external_metric)
        self.y_data = self.y_scaler.transform(self.external_metric)

        assert len(ranked_knobs_info) == len(external_metric)
    
    def __len__(self):
        return len(self.ranked_knobs_info)

    def __getitem__(self,idx):
        return self.X_data[idx],self.y_data[idx]
