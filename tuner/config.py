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

    def save_results(self,taget_workload, best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_path, log_dir):
        self.target = taget_workload
        self.best_epoch = best_epoch
        self.best_th_mse = best_th_loss
        self.best_la_mse = best_la_loss
        self.best_th_mae = best_th_mae_loss
        self.best_la_mae = best_la_mae_loss
        self.model_path = model_path
        self.log_dir = log_dir
