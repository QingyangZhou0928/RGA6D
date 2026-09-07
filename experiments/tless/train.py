# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------

import time
import copy

import torch

from models.engine.iter_based_trainer import IterBasedTrainer
from models.utils.torch import build_warmup_cosine_lr_scheduler

from config_train import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import CoarselLoss, FineLoss, Evaluator

class Trainer(IterBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_iteration=cfg.optim.max_iteration, snapshot_steps=cfg.optim.snapshot_steps)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)

        # ===================== freeze partial backbone =====================
        freeze_keys = [
        "encoder1_1",
        "encoder1_2",
        "encoder2_1",
        ]
        for name, p in model.named_parameters():
            if any(k in name for k in freeze_keys):
                p.requires_grad = False

        print("Frozen backbone")

        # ===================== freeze partial backbone =====================
        low_lr = []
        high_lr = []
        region_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if "region" in name.lower():
                region_params.append(p)

            elif any(k in name for k in [
                "encoder2_2",
                "encoder2_3",
                "encoder3",
            ]):
                low_lr.append(p)

            else:
                high_lr.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": low_lr, "lr": cfg.optim.lr*0.1},
                {"params": region_params, "lr": cfg.optim.lr*0.1},  
                {"params": high_lr, "lr": cfg.optim.lr},
            ],
            weight_decay=cfg.optim.weight_decay,
        )
       
        self.register_optimizer(optimizer)
        scheduler = build_warmup_cosine_lr_scheduler(
            optimizer,
            total_steps=cfg.optim.max_iteration,
            warmup_steps=cfg.optim.warmup_steps,
            eta_init=cfg.optim.eta_init,
            eta_min=cfg.optim.eta_min,
            grad_acc_steps=cfg.optim.grad_acc_steps,
        )
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_coarse = CoarselLoss(cfg).cuda()
        self.loss_fine = FineLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, iteration, data_dict):
        loss_dict = {}
        # coarse
        output_dict_coarse = self.model(data_dict, {})
        loss_dict_coarse = self.loss_coarse(output_dict_coarse, data_dict)
        loss_dict.update(loss_dict_coarse)
        loss_dict_coarse['coarse_loss'].backward()  # backward & optimization
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        output_dict = {k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) 
               for k,v in output_dict_coarse.items()}

        # ===================== freeze partial backbone =====================
        for name, p in self.model.named_parameters():
            if any(k in name for k in [
                "encoder2_2",
                "encoder2_3",
                "encoder3",
                ]):     
                p.grad = None
                p.requires_grad = False

        # fine
        N_fine = output_dict['coarse_est_transform'].shape[0] 
        for group_id in range(N_fine):
            output_dict,success_flag = self.model(data_dict, output_dict, group_id)
            if success_flag:
                loss_dict_fine = self.loss_fine(output_dict, data_dict)
                for k, v in loss_dict_fine.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    if k in loss_dict:  
                        if isinstance(loss_dict[k], list):
                            loss_dict[k].append(val)
                        else:
                            loss_dict[k] = [loss_dict[k], val]
                    else: 
                        loss_dict[k] = [val]
                group_fine_loss = loss_dict_fine['f_loss'] / N_fine  
                group_fine_loss.backward()  # backward & optimization
            torch.cuda.empty_cache()

        # ===================== freeze partial backbone =====================
        for name, p in self.model.named_parameters():
            if any(k in name for k in [
        "encoder2_2",
        "encoder2_3",
        "encoder3",
        ]):
                p.requires_grad = True

        result_dict = self.evaluator(output_dict, data_dict)  
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, iteration, data_dict):
        loss_dict = {}
        # coarse
        output_dict = self.model(data_dict, {})
        loss_dict_coarse = self.loss_coarse(output_dict, data_dict)
        loss_dict.update(loss_dict_coarse)

        torch.cuda.empty_cache()

        # fine
        N_fine = output_dict['coarse_est_transform'].shape[0] 
        for group_id in range(N_fine):
            output_dict, success_flag = self.model(data_dict, output_dict, group_id)
            if success_flag:
                loss_dict_fine = self.loss_fine(output_dict, data_dict)
                for k, v in loss_dict_fine.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    if k in loss_dict: 
                        if isinstance(loss_dict[k], list):
                            loss_dict[k].append(val)
                        else:
                            loss_dict[k] = [loss_dict[k], val]
                    else:  
                        loss_dict[k] = [val]
    
            torch.cuda.empty_cache()

        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()