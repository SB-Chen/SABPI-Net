import os
import os.path as osp
from typing import Dict, Optional
import torch

def save_checkpoints_w_rules(args,
                             save_model_file:Dict,
                             current_epoch:int, # current_epoch >= 1
                             logger,
                             best_rule:Optional[Dict[str, float]] = None,
                             save_w_epoch_frequency=False,
                             other_rule:Optional[Dict[str, float]] = None,
                             save_current=True):
   
    if save_w_epoch_frequency:
        epoch_save_name = f"{args.dataset_name}_{args.model_name}_epoch_{str(current_epoch)}_{args.train_start_time}.pth"
        if current_epoch % args.save_model_fre_epoch == 0 or current_epoch <= 5:
            save_name = (f"fold{args.cross_validation_fold}_" + epoch_save_name) if args.cross_validation_fold is not None else epoch_save_name
            if (args.train_distributed and args.rank == 0) or not args.train_distributed:
                torch.save(save_model_file, osp.join(args.model_params_path, save_name))
                logger.info(f"Save model [Epoch:{current_epoch}]")

    
    if best_rule is not None:
        key = '_'.join(f"{k}" for k in list(best_rule.keys()))
        best_save_name = f"{args.dataset_name}_{args.model_name}_best_{key}_{args.train_start_time}.pth"
        save_name = (f"fold{args.cross_validation_fold}_" + best_save_name) if args.cross_validation_fold is not None else best_save_name
        if (args.train_distributed and args.rank == 0) or not args.train_distributed:
            torch.save(save_model_file, osp.join(args.best_model_weight_path, save_name))
            best_rule_info = ','.join(f"{k}:{str(v)}" for k, v in best_rule.items())
            logger.info(f"Save best model [Epoch:{current_epoch}, {best_rule_info}]")

    if other_rule is not None:
        other_rule_info = '_'.join(f"{k}:{str(v)}" for k, v in other_rule.items())
        other_save_name = f"{args.dataset_name}_{args.model_name}_{str(current_epoch)}_{other_rule_info}_{args.train_start_time}.pth"
        save_name = (f"fold{args.cross_validation_fold}_" + other_save_name) if args.cross_validation_fold is not None else other_save_name
        if (args.train_distributed and args.rank == 0) or not args.train_distributed:
            torch.save(save_model_file, osp.join(args.model_params_path, save_name))
            other_rule_info = ','.join(f"{k}:{str(v)}" for k, v in other_rule.items())
            logger.info(f"Save model [Epoch:{current_epoch}, {other_rule_info}]")

    
    if save_current:
        current_save_name = f"{args.dataset_name}_{args.model_name}_final_epoch_{args.train_start_time}.pth"
        save_name = (f"fold{args.cross_validation_fold}_" + current_save_name) if args.cross_validation_fold is not None else current_save_name
        if (args.train_distributed and args.rank == 0) or not args.train_distributed:
            torch.save(save_model_file, osp.join(args.final_model_weight_path, save_name))
            logger.info(f"Save current model [Epoch:{current_epoch}]")

