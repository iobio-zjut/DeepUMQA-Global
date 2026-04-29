import pytorch_lightning as pl
import torch
import numpy
from structure_rank.Model.ModelNet import ModelRank
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class ScoreLoss(torch.nn.Module):
    def __init__(self, config):
        super(ScoreLoss, self).__init__()
        self.config = config
        self.Esto_Loss = torch.nn.CrossEntropyLoss()
        self.Mask_Loss = torch.nn.BCEWithLogitsLoss()
        self.Lddt_Loss = torch.nn.MSELoss()
        self.loss_weight = config.loss_weight

    def loss(self, out, batch):
        losses = {}
        dev_pred, mask_pred, lddt_pred, (dev_logits, mask_logits) = out
        dev, dev_1hot, mask = batch["deviation"], batch["deviation_1hot"], batch["mask"]
        dev_true = dev[0]
        dev_1hot_true = dev_1hot[0]
        mask_true = mask[0]
        dev_loss = self.Esto_Loss(dev_logits, dev_true.long())
        mask_loss = self.Mask_Loss(mask_logits, mask_true)
        lddt_loss = self.Lddt_Loss(lddt_pred, lddt_true.float())
        loss = self.loss_weight[0]*dev_loss + self.loss_weight[1]*mask_loss + self.loss_weight[2]*lddt_loss
        losses["total_loss"] = loss
        losses["dev_loss"] = dev_loss
        losses["mask_loss"] = mask_loss
        losses["lddt_loss"] = lddt_loss
        return losses

    def forward(self, out, batch):
        return self.loss(out, batch)

class QualityLoss(torch.nn.Module):
    def __init__(self):
        super(QualityLoss, self).__init__()
        self.bce_Loss = torch.nn.BCEWithLogitsLoss()

    def loss(self, quality_logits, batch):
        losses = {}
        quality_true = batch['quality']
        mask_loss = self.bce_Loss(quality_logits, quality_true)
        losses["total_loss"] = mask_loss
        return losses
    def forward(self, out, batch):
        return self.loss(out, batch)

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.Loss = torch.nn.MSELoss()
    def loss(self, quality_logits, batch):
        losses = {}           
        quality_true = batch['quality'].float()
        loss = self.Loss(quality_logits, quality_true) * 1
        losses["total_loss"] = loss
        return losses
    def forward(self, out, batch):
        return self.loss(out, batch)

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.Loss = torch.nn.MSELoss()
        self.weight = weight
    
    def loss(self, quality_logits, batch):
        losses = {}           
        quality_true = batch['quality'].float()
        loss = self.Loss(quality_logits, quality_true)
        weighted_loss = loss * self.weight
        losses["total_loss"] = weighted_loss
        return losses
    
    def forward(self, out, batch):
        return self.loss(out, batch)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.Loss = torch.nn.MSELoss()
    def loss(self, quality_logits, batch):
        losses = {}            
        quality_true = batch['quality'].float()
        mse_loss = self.Loss(quality_logits, quality_true) * 1
        rmse_loss = torch.sqrt(mse_loss)
        losses["total_loss"] = rmse_loss
        return losses
    def forward(self, out, batch):
        return self.loss(out, batch)

class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def loss(self, quality_logits, batch):
        losses = {}           
        quality_true = batch['quality'].float()
        loss = torch.nn.functional.l1_loss(quality_logits, quality_true) # 使用 F.l1_loss 计算 MAE
        losses["total_loss"] = loss
        return losses
    def forward(self, out, batch):
        return self.loss(out, batch)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def loss(self, quality_logits, batch):
        losses = {} 
        quality_true = batch['quality'].float()
        error = quality_logits - quality_true
        loss = torch.log(torch.cosh(error))
        losses["total_loss"] = loss
        return losses
    def forward(self, quality_logits, batch):
        return self.loss(quality_logits, batch)

class QuantileLoss(torch.nn.Module):
    def __init__(self, tau):
        super(QuantileLoss, self).__init__()
        self.tau = tau

    def loss(self, quality_logits, batch):
        losses = {} 
        quality_true = batch['quality'].float()
        error = (quality_logits - quality_true) ** 2
        loss = torch.where(error >= 0, self.tau * error, (1 - self.tau) * error)
        losses["total_loss"] = loss
        return losses
       

    def forward(self, quality_logits, batch):
        return self.loss(quality_logits, batch)

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.Loss = torch.nn.CrossEntropyLoss()

    def loss(self, quality_logits, batch):
        losses = {}
        quality_true = batch['quality']
        losses["total_loss"] = loss
        return losses
    def forward(self, out, batch):
        return self.loss(out, batch)

class MultimerModule(pl.LightningModule):
    def __init__(self, args):
        super(MultimerModule, self).__init__()
       
        self.model = ModelRank()
        if args.loss_type.lower()=='mse':
            self.loss = MSELoss()
        elif args.loss_type.lower()=='rmse':
            self.loss = RMSELoss()
        elif args.loss_type.lower()=='mae':
            self.loss = MAELoss()
        elif args.loss_type.lower()=='wmse':
            self.loss = WeightedMSELoss(5)
        elif args.loss_type.lower()=='logcosh':
            self.loss = LogCoshLoss()
        elif args.loss_type.lower()=='quantile':
            self.loss =QuantileLoss(tau=0.75)
        self.step = 0
        self.epoch = 0
        self.incomplete_res=0
        self.config = args

    def forward(self, data):
        
        # if data['_1d'].shape[-1]<3:
        #     return None 

        return self.model(data)
        # return None

    
    def training_step(self, batch, batch_idx):
        if len(list(batch.keys())) == 1:
            self.incomplete_res+=1
            # print(batch["name"],self.incomplete_res)
            return None
        
        
        outputs = self(batch)
        # if outputs is None:
        #     return None
         
        loss = self.loss(outputs, batch)
        self.log(
            "train/total_loss", loss["total_loss"], on_step=True, on_epoch=True, logger=True
        )
        if 'dev_loss' in loss:
            self.log(
                "train/dev_loss", loss["dev_loss"], on_step=True, on_epoch=True, logger=True
            )
        if 'mask_loss' in loss:
            self.log(
                "train/mask_loss", loss["mask_loss"], on_step=True, on_epoch=True, logger=True
            )
        if 'lddt_loss' in loss:
            self.log(
                "train/lddt_loss", loss["lddt_loss"], on_step=True, on_epoch=True, logger=True
            )
        return loss["total_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, eps=self.config.eps
        )
        opt = {
            "optimizer": optimizer,
        }
        return opt
    
    def validation_step(self, batch, batch_idx):
        if len(list(batch.keys())) == 1:
            self.incomplete_res+=1
            return None
        
        outputs = self(batch)
        loss = self.loss(outputs, batch)
        
        self.log(
            "val_loss", loss["total_loss"], on_step=True, on_epoch=True, logger=True
        )
        
        return loss["total_loss"]
    
    def set_save_ck(self, checkpoint_dir):
        every_epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='every_epoch-{epoch:02d}',
        every_n_epochs=5, 
        auto_insert_metric_name=False, 
        save_top_k=-1
        )  # save 3 epoch

        step_epoch_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='latest_step-{step:06d}',
        every_n_train_steps=3, 
        auto_insert_metric_name=False, 
        save_top_k=1
        )  # save the newest step

        val_best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='val_best-{epoch:02d}-{val_loss:.5f}',
        monitor='val_loss',   
        save_top_k=3,  
        mode='min'  
        )   # save val best 3

        # early_stopping_callback = EarlyStopping(
        # monitor='val_loss', 
        # min_delta=0.001, 
        # patience=8, 
        # mode='min'  
        # ) 

        model_callbacks = [every_epoch_checkpoint, step_epoch_checkpoint, val_best_checkpoint]
        return model_callbacks

    # def on_after_backward(self):
    #     torch.cuda.empty_cache() 
    

       
              
    