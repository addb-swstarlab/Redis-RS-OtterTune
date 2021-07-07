class Config():
    def __init__(self,persistence,db,cluster,rki,top_k,model_mode,n_epochs,lr):
        self.persistence = persistence
        self.db = db
        self.cluster = cluster
        self.rki = rki
        self.topk = top_k
        self.model_mode = model_mode
        self.n_epochs = n_epochs
        self.lr = lr

    def save_results(self,taget_workload, best_epoch, best_mse, best_mae, model_path, log_dir):
        self.target = taget_workload
        self.best_epoch = best_epoch
        self.best_mse = best_mse
        self.best_mae = best_mae
        self.model_path = model_path
        self.log_dir = log_dir
