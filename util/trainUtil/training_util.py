from typing import List
import os.path as osp
import os
import numpy as np
import torch.nn.functional as F
import logging
import sys
from model.get_model import getModel
from util.model_optimizer import create_optimizer
from util.lr_scheduler import create_lr_scheduler
from util.loss_func import createLossFunc

from util.trainUtil.utils import save_checkpoints_w_rules
from tqdm import tqdm  
from sklearn.metrics import confusion_matrix
import timeit  
import torch
import torch.utils.data as torchdata
import time
import wandb
from util.metric import (calculate_metric_for_binary_label, 
                         calculate_metric_for_multi_label,
                         calculate_metric_for_each_label)

from util.trainUtil.distributed_utils import reduce_value, is_main_process, gather_list


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


class TrainUtil:
    def __init__(self,
                 args,
                 # dataset_name: str,
                 img_path: str,
                 num_classes: int,
                 img_size: int,
                 data_excel_file_path,
                 train_excel_sheet_name: str,
                 val_excel_sheet_name: str,
                 excel_column_name: List[str],
                 get_data,
                 cross_validation_fold=None,
                 train_transforms=None,
                 test_transforms=None,
                 **kwargs):

        self.args = args
        self.args.num_classes = num_classes
        self.args.img_size = img_size
        self.args.train_transforms = train_transforms
        self.args.test_transforms = test_transforms
        # self.args.dataset_name = dataset_name
        self.args.img_path = img_path
        self.args.data_excel_file_path = data_excel_file_path
        self.args.train_excel_sheet_name = train_excel_sheet_name
        self.args.val_excel_sheet_name = val_excel_sheet_name
        self.args.excel_column_name = excel_column_name
        self.args.get_data = get_data
        self.args.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.args.train_start_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        self.args.cross_validation_fold = str(cross_validation_fold) if isinstance(cross_validation_fold, int) \
            else cross_validation_fold

        if 'classes_weight' in kwargs:
            self.args.classes_weight = kwargs['classes_weight']
        if 'resume' in kwargs:
            # Load existing model parameters and start training
            self.args.resume = kwargs['resume']
        # Long-tailed option
        if 'head_class_idx' in kwargs:
            self.args.head_class_idx = kwargs['head_class_idx']
        if 'med_class_idx' in kwargs:
            self.args.med_class_idx = kwargs['med_class_idx']
        if 'tail_class_idx' in kwargs:
            self.args.tail_class_idx = kwargs['tail_class_idx']
        if 'rho' in kwargs:
            self.args.rho = kwargs['rho']
        if 'data_func_flag' in kwargs:
            self.args.data_func_flag = kwargs['data_func_flag']
        if 'class_balanced_data' in kwargs:
            self.args.class_balanced_data = kwargs['class_balanced_data']

        self.args.best_model_weight_path = osp.join(self.args.train_record_save_root,
                                                    self.args.model_name + '_' + args.exp_str, 'best_model_weight')
        self.args.final_model_weight_path = osp.join(self.args.train_record_save_root,
                                                     self.args.model_name + '_' + args.exp_str, 'final_model_weight')
        self.args.model_params_path = osp.join(self.args.train_record_save_root,
                                               self.args.model_name + '_' + args.exp_str, 'model_weight')

        if not os.path.isdir(self.args.best_model_weight_path):
            os.makedirs(self.args.best_model_weight_path, exist_ok=True)
        if not os.path.isdir(self.args.final_model_weight_path):
            os.makedirs(self.args.final_model_weight_path, exist_ok=True)
        if not os.path.isdir(self.args.model_params_path):
            os.makedirs(self.args.model_params_path, exist_ok=True)

        self.args.train_log_file_path = osp.join(self.args.train_record_save_root,
                                                 self.args.model_name + '_' + args.exp_str, 'train_log_file')
        if not os.path.isdir(self.args.train_log_file_path):
            os.makedirs(self.args.train_log_file_path, exist_ok=True)

        if (not args.train_distributed) or (args.train_distributed and args.rank == 0):
            wandb_dir = osp.join(self.args.train_record_save_root, 'wandb_dir')
            if not os.path.isdir(wandb_dir):
                os.makedirs(wandb_dir, exist_ok=True)
            wandb_name = f"{self.args.model_name}_{args.exp_str}_{self.args.cross_validation_fold}_{self.args.train_start_time}" \
                if self.args.cross_validation_fold is not None \
                else f"{self.args.model_name}_{args.exp_str}_{self.args.train_start_time}"

            if not os.path.isdir(osp.join(wandb_dir, wandb_name)):
                os.makedirs(osp.join(wandb_dir, wandb_name))
            wandb.init(
                # mode="offline",
                project=kwargs['wandb_project_name'] if 'wandb_project_name' in kwargs else self.args.dataset_name,
                entity="ethansbchen",
                name=wandb_name,
                dir=osp.join(wandb_dir, wandb_name)  # wandb_dir
            )
            wandb.config.update(self.args)

    @staticmethod
    def trainlog(logger_file_path, logger_name, head='%(message)s'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        fHandler = logging.FileHandler(logger_file_path, mode='w')
        fHandler.setLevel(logging.INFO)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)

        return logger

    @staticmethod
    def load_data_and_prepare(args, load_data):
        """
        Load the data and transfer it to the specified device.
        """
        img_data, img_hfc, img_data_changed_lf, img_data_changed_lf_hfc, img_label, img_name = load_data
        img_hfc = img_hfc.to(args.device)
        img_data_changed_lf = img_data_changed_lf.to(args.device)
        img_data_changed_lf_hfc = img_data_changed_lf_hfc.to(args.device)
        return img_data, img_label, img_name, {'img_hfc': img_hfc,
                                               'img_data_changed_lf': img_data_changed_lf,
                                               'img_data_changed_lf_hfc': img_data_changed_lf_hfc}

    def compute_loss_and_predict(self, args, model, img_data, img_label, extra_data, epoch, model_optimizer, loss_fuc):
        output, x_embedded_fea, x_2_embedded_fea = model(
                        img_data,
                        extra_data['img_hfc'],
                        extra_data['img_data_changed_lf'],
                        extra_data['img_data_changed_lf_hfc']
                    )
        loss_ce = loss_fuc(output, img_label)
        cosine_sim = F.cosine_similarity(x_embedded_fea, x_2_embedded_fea, dim=1)
        loss_cosine_sim = 1 - cosine_sim.mean()
        loss = loss_ce + 0.1 * loss_cosine_sim  # 0.01
        _, pred_classes = torch.max(output, 1)

        predicted_label_correct = torch.eq(pred_classes, img_label).sum()
        return loss, pred_classes, predicted_label_correct

    @torch.no_grad()
    def evaluation(self, args, model, data_loader, loss_func, epoch):
        model.eval()  # Set model to evaluate mode
        pred_label = []
        true_label = []
        label_probability = []
        sample_num = torch.zeros(1).to(args.device)
        step = 0
        pred_correct = torch.zeros(1).to(args.device)
        loss_accumulate = torch.zeros(1).to(args.device)
        if (args.train_distributed and is_main_process()) or not args.train_distributed:
            data_loader = tqdm(data_loader, file=sys.stdout)

        for batch_cnt, load_data in enumerate(data_loader):
            img_data, img_hfc, img_label, img_name = load_data
            img_hfc = img_hfc.to(args.device)
            step += 1
            img_data, img_label = img_data.to(args.device), img_label.to(args.device)
            sample_num += img_data.shape[0]
            output = model(img_data, img_hfc)
            # print(f"evaluate-->img_data:{img_data.shape}, output:{output.shape}")
            _, pred_classes = torch.max(output, 1)
            loss = loss_func(output, img_label)
            m = torch.nn.Softmax(dim=1)
            softmax_outputs = m(output)
            pred_correct += torch.eq(pred_classes, img_label).sum()
            if args.train_distributed:
                loss = reduce_value(loss, average=True)
            loss_accumulate += loss

            pred_label.extend(pred_classes.cpu().numpy())
            true_label.extend(img_label.cpu().numpy())  # img_label.cpu().data.numpy()

            label_probability.extend(softmax_outputs.data.cpu().numpy().tolist())

        # Wait for all processes to finish computing.
        if args.train_distributed:
            if args.device != torch.device("cpu"):
                torch.cuda.synchronize(args.device)

        if args.train_distributed:
            true_label = gather_list(true_label)
            true_label = true_label.tolist()
            pred_label = gather_list(pred_label)
            pred_label = pred_label.tolist()
            label_probability = gather_list(label_probability)
            label_probability = label_probability.tolist()

        result_confusion_matrix = confusion_matrix(true_label, pred_label)

        if args.num_classes == 2:
            metric_result_info = calculate_metric_for_binary_label(pred_label=pred_label,
                                                                   gt_label=true_label,
                                                                   label_probability=label_probability)
        else:
            metric_result_info = calculate_metric_for_multi_label(num_classes=args.num_classes,
                                                                  pred_label=pred_label,
                                                                  gt_label=true_label,
                                                                  label_probability=label_probability)

        single_class_metric = calculate_metric_for_each_label(multi_label_confusion_matrix=result_confusion_matrix)
        if (args.train_distributed and args.rank == 0) or (not args.train_distributed):
            print(f"Evaluate results:")
            print('confusion_matrix:\n', metric_result_info['confusion_matrix'])
            # 将所有的key和value拼接成一个字符串，但排除'confusion_matrix'
            metric_output = ', '.join(
                f"{k}: {round(v * 100, 2)}" for k, v in metric_result_info.items() if k not in ['confusion_matrix'])
            print(f"overall metric:{metric_output}")

            for label, metric_dict in enumerate(single_class_metric):
                metric_output = ', '.join(f"{k}: {round(v * 100, 2)}" for k, v in metric_dict.items())
                print(f"label:{label}, {metric_output}")

            if all(hasattr(args, attr) for attr in ['head_class_idx', 'med_class_idx', 'tail_class_idx']):
                # print("[INFORMATION] The size of the cf is", result_confusion_matrix.shape)
                cls_cnt = result_confusion_matrix.sum(axis=1)
                cls_hit = np.diag(result_confusion_matrix)  # num of correct preds
                cls_acc = cls_hit / cls_cnt

                head_acc = cls_acc[args.head_class_idx].mean() * 100
                med_acc = cls_acc[args.med_class_idx].mean() * 100
                tail_acc = cls_acc[args.tail_class_idx].mean() * 100
                print(f"The head Sen is {head_acc},")
                print(f"The med Sen is {med_acc},")
                print(f"The tail Sen is {tail_acc}")
                if (not args.train_distributed) or (args.train_distributed and args.rank == 0):
                    wandb.log({'head_sen': head_acc, 'med_sen': med_acc, 'tail_sen': tail_acc, 'epoch': epoch})
                out_cls_acc = 'Class Sen: %s' % (
                    np.array2string(cls_acc * 100, separator=',', formatter={'float_kind': lambda x: "%.2f" % x}))
                print(out_cls_acc)

        if args.train_distributed:
            pred_correct = reduce_value(pred_correct, average=False)
            sample_num = reduce_value(sample_num, average=False)

        acc_evaluate = round((pred_correct.item() / sample_num.item()) * 100, 4)
        return round(loss_accumulate.item() / step,
                     4), acc_evaluate, metric_result_info, result_confusion_matrix, single_class_metric

    def train_one_epoch(self, args, logger, model, epoch,
                        model_optimizer, loss_fuc, lr_scheduler, train_loader):
        model.train(True)
        if args.train_distributed:
            sample_num = torch.zeros(1).to(args.device)
        else:
            sample_num = 0

        step = 0
        pred_correct = torch.zeros(1).to(args.device)
        loss_accumulate = torch.zeros(1).to(args.device)
        # mean_loss = torch.zeros(1).to(args.device)
        model_optimizer.zero_grad()
        start = timeit.default_timer()  # start timer
        if (args.train_distributed and is_main_process()) or not args.train_distributed:
            log_info = f"Start train {args.model_name} for {args.dataset_name}, fold{args.cross_validation_fold} epoch:{epoch}-->" if args.cross_validation_fold is not None \
                else f"Start train {args.model_name} for {args.dataset_name},  epoch:{epoch}-->"
            logger.info(log_info)
            train_loader = tqdm(train_loader, file=sys.stdout)
        for batch_cnt, load_data in enumerate(train_loader):
            # load data
            img_data, img_label, img_name, extra_data = self.load_data_and_prepare(args, load_data)
            # load data End!
            step += 1
            img_data, img_label = img_data.to(args.device), img_label.to(args.device)
            sample_num += img_data.shape[0]
            # get model output and calculate loss
           
            loss, pred_classes, predicted_label_correct = self.compute_loss_and_predict(args=args,
                                                                                        model=model,
                                                                                        img_data=img_data,
                                                                                        img_label=img_label,
                                                                                        extra_data=extra_data,
                                                                                        epoch=epoch,
                                                                                        model_optimizer=model_optimizer,
                                                                                        loss_fuc=loss_fuc)
            pred_correct += predicted_label_correct
            loss.backward()
            model_optimizer.step()
            # wandb.log({'batch_loss': loss, 'epoch': epoch})
            if args.train_distributed:
                loss = reduce_value(loss, average=True)
            # mean_loss = (mean_loss * batch_cnt + loss.detach()) / (batch_cnt + 1)  # update mean losses
            loss_accumulate += loss.detach()

            if step % 100 == 0:
                if (args.train_distributed and is_main_process()) or not args.train_distributed:
                    acc_value = (reduce_value(pred_correct, average=False).item() / reduce_value(sample_num,
                                                                                                 average=False).item()) \
                        if (args.train_distributed and is_main_process()) else pred_correct.item() / sample_num
                    logger.info(f"\tEpoch:{epoch}-->Batch({step}/{len(train_loader)}) done. "
                                f"Current loss:{round(loss.item(), 4)},"  # loss_accumulate.item() / step
                                f"acc:{round(acc_value * 100, 2)},"
                                f"lr:{model_optimizer.param_groups[-1]['lr'] if args.model_optimizer == 'SAM' else lr_scheduler.get_last_lr()[0]}")

            # wandb.log({'batch_lr': model_optimizer.param_groups[-1]['lr'] if args.model_optimizer == 'SAM' else lr_scheduler.get_last_lr()[0],
            #            'epoch': epoch})
            if args.lr_scheduler in ['step', 'cos_lr_step']:
                lr_scheduler.step()
            # zero the parameter gradients
            model_optimizer.zero_grad()

        if args.lr_scheduler in ['cos_lr_epoch']:
            lr_scheduler.step()

        # Wait for all processes to finish computing.
        if args.train_distributed:
            if args.device != torch.device("cpu"):
                torch.cuda.synchronize(args.device)

        # Record the average loss and accuracy after completing an epoch of training.
        train_loss = loss_accumulate.item() / step
        end = timeit.default_timer()
        if args.train_distributed:
            pred_correct = reduce_value(pred_correct, average=False)
            sample_num = reduce_value(sample_num, average=False)
            train_acc = pred_correct.item() / sample_num.item()
            if is_main_process():
                logger.info(f"Train {args.model_name} for {args.dataset_name}, epoch:{epoch} done. "
                            f"acc:{round(train_acc * 100, 4)},"
                            # f"mean loss:{round(mean_loss.item(), 4)},"
                            f"loss:{round(train_loss, 4)},"
                            f"lr:{round(lr_scheduler.get_last_lr()[0], 4)}, "
                            f"run {round(end - start, 2)}s")
        else:
            train_acc = pred_correct.item() / sample_num
            logger.info(f"Train {args.model_name} for {args.dataset_name}, epoch:{epoch} done. "
                        f"acc:{round(train_acc * 100, 4)}, "
                        # f"mean loss:{round(mean_loss.item(), 4)},"
                        f"loss:{round(train_loss, 4)},"
                        f"lr:{round(lr_scheduler.get_last_lr()[0], 4)},"
                        f"run {round(end - start, 2)}s")
            if (not args.train_distributed) or (args.train_distributed and args.rank == 0):
                wandb.log({'epoch_loss': train_loss,
                           'epoch_train_acc': train_acc,
                           'epoch_lr': lr_scheduler.get_last_lr()[0],
                           'epoch': epoch})

        return train_loss, lr_scheduler.get_last_lr()[0]  # model_optimizer.param_groups[0]["lr"]

    def running(self):
        args = self.args
        args.num_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])  # number of workers
        train_excel_file_info = {'file_path': args.data_excel_file_path,
                                 'sheet_name': args.train_excel_sheet_name,
                                 'column_name': args.excel_column_name}
        val_excel_file_info = {'file_path': args.data_excel_file_path,
                               'sheet_name': args.val_excel_sheet_name,
                               'column_name': args.excel_column_name}

        train_data = args.get_data(excel_file_info=train_excel_file_info,
                                    img_path=args.img_path,
                                    img_size=args.img_size,
                                    train_flag=True,
                                    transform=args.train_transforms,
                                    class_balanced_data=args.class_balanced_data if hasattr(args,
                                                                                            'class_balanced_data') else False,
                                    args=args)
        if (hasattr(args,
                    'data_func_flag') and args.data_func_flag == 'LongTailed') or args.train_rule == 'Reweight':
            args.cls_num_list = train_data.cls_num_list
            print(f"The class number list for {args.dataset_name} is ", args.cls_num_list)

        val_data = args.get_data(excel_file_info=val_excel_file_info,
                                    img_path=args.img_path,
                                    img_size=args.img_size,
                                    train_flag=False,
                                    transform=args.test_transforms,
                                    class_map=train_data.class_map if (hasattr(args, 'data_func_flag') and args.data_func_flag == 'LongTailed') else None,
                                    args=args)

        # cls_num_list is used for class weight calculation.
        args.cls_num_list = train_data.cls_num_list
        logger_name = f"{args.dataset_name}_{args.model_name}_{args.train_start_time}_train_log"
        if args.train_distributed:
            logger_name += f"_rank{args.rank}"
        if args.cross_validation_fold is not None:
            logger_name += f"_fold{args.cross_validation_fold}"
        logger = self.trainlog(logger_file_path=osp.join(args.train_log_file_path, logger_name + '.txt'),
                               logger_name=logger_name)
        print(f"create logger!")
        
        if args.train_distributed:
            args.device = torch.device('cuda')
            args.base_learning_rate *= args.world_size  # The learning rate is scaled according to the number of parallel GPUs.
            print('===> Distributed Loading datasets')
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
            
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
            if args.rank == 0:
                print('Using {} dataloader workers every process'.format(args.num_worker))
            train_data_loader = torch.utils.data.DataLoader(train_data,
                                                            batch_sampler=train_batch_sampler,
                                                            # shuffle=True,
                                                            # pin_memory = True,
                                                            # collate_fn=train_data.collate_fn,
                                                            num_workers=args.num_worker)

            val_data_loader = torch.utils.data.DataLoader(val_data,
                                                          batch_size=args.batch_size,
                                                          sampler=val_sampler,
                                                          shuffle=False,
                                                          # pin_memory=True,
                                                          num_workers=args.num_worker,
                                                          # collate_fn=val_data_set.collate_fn
                                                          )
        else:
            print('===> Loading datasets')
            train_data_loader = torch.utils.data.DataLoader(train_data,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_worker)
            val_data_loader = torch.utils.data.DataLoader(val_data,
                                                          batch_size=args.batch_size,
                                                          shuffle=False,
                                                          num_workers=args.num_worker)
           
        model = getModel(args, pre_trained=args.pre_trained)
        model = model.to(args.device)
        model_optimizer = create_optimizer(args, model.parameters())
        lr_scheduler = create_lr_scheduler(model_optimizer, args, num_step=len(train_data_loader))

        start_epoch = 1
        acc_best = 0.0
        if hasattr(args, 'resume') and args.resume is not None:
            print(f"Load resume model param")
            checkpoint = torch.load(args.resume, map_location='cpu')
            acc_best = checkpoint['acc']
            print(f"resume param best acc:{acc_best}")
            model.load_state_dict(checkpoint['model'])
            model_optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        if args.train_distributed:
            if args.syncBN:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)
           
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.gpu],
                                                              find_unused_parameters=True)  # , find_unused_parameters=True

        if args.model_name in ["SBAINet_tfs"]:
            # For Tail-aware feature fusion via fine-tuning
            # Freeze all parameters except the classification layer
            for name, param in model.named_parameters():
                if "head" not in name:  # Only train the classification layer
                    param.requires_grad = False
            # Set optimizer (optimize only the classification layer)
            model_optimizer = create_optimizer(args, model.head.parameters())
            lr_scheduler = create_lr_scheduler(model_optimizer, args, num_step=len(train_data_loader))
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

  
        train_result_metric = {}
        if hasattr(args, 'tail_class_idx') and hasattr(args, 'med_class_idx'):
            f1_best = 0.0

        for epoch in range(start_epoch, args.epoch + 1):
            if args.train_distributed:
                train_sampler.set_epoch(epoch)

            loss_func = createLossFunc(args) if args.train_rule == 'None' else createLossFunc(args, epoch)

            loss_one_epoch, lr_one_epoch = self.train_one_epoch(args=args,
                                                                logger=logger,
                                                                model=model,
                                                                epoch=epoch,
                                                                model_optimizer=model_optimizer,
                                                                loss_fuc=loss_func,
                                                                lr_scheduler=lr_scheduler,
                                                                train_loader=train_data_loader)

            if epoch % args.evaluation_Frequency == 0:
                t0 = time.time()
                (val_loss,
                 val_acc,
                 metric_result_info,
                 result_confusion_matrix,
                 single_class_metric) = self.evaluation(args=args, model=model,
                                                        data_loader=val_data_loader,
                                                        loss_func=loss_func,
                                                        epoch=epoch)
                if (not args.train_distributed) or (args.train_distributed and args.rank == 0):
                    wandb.log({'epoch_val_loss': val_loss,
                               'epoch_val_acc': val_acc,
                               'epoch': epoch})
                
                if len(list(train_result_metric.keys())) == 0:
                    for metric, _ in metric_result_info.items():
                        if metric != 'confusion_matrix':
                            train_result_metric[metric] = []

                for metric, metric_value in metric_result_info.items():
                    if metric != 'confusion_matrix':
                        train_result_metric[metric].append(metric_value)

                t1 = time.time()

                # save model
      
                save_file = {
                    "state_dict": model.state_dict() if not args.train_distributed else model.module.state_dict(),
                    "optimizer": model_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "acc": acc_best}
                # Save the current model, and save the model at a specified frequency.
                save_checkpoints_w_rules(args=args, save_model_file=save_file,
                                         current_epoch=epoch, logger=logger,
                                         save_current=True, save_w_epoch_frequency=True)
                # Save the best model in val
                if val_acc >= acc_best:
                    acc_best = val_acc
                    best_rule = {'val_acc': round(val_acc, 2)}
                    save_checkpoints_w_rules(args=args, save_model_file=save_file,
                                             current_epoch=epoch, logger=logger,
                                             save_current=False, best_rule=best_rule)
                if hasattr(args, 'tail_class_idx') and hasattr(args, 'med_class_idx'):
                    all_class_indices = args.tail_class_idx + args.med_class_idx
                    val_f1 = sum(round(single_class_metric[idx]["F1"] * 100, 2) for idx in all_class_indices) / len(
                        all_class_indices)
                    if val_f1 >= f1_best:
                        f1_best = val_f1
                        best_rule = {'val_F1': val_f1}
                        save_checkpoints_w_rules(args=args, save_model_file=save_file,
                                                 current_epoch=epoch, logger=logger,
                                                 save_current=False, best_rule=best_rule)
    
                if (args.train_distributed and args.rank == 0) or not args.train_distributed:
                    logger.info(
                        f"Eval after epoch: {epoch}. acc:{val_acc}, loss:{val_loss}, best val acc:{acc_best},"
                        f"lr:{lr_scheduler.get_last_lr()[0]}, run {t1 - t0}s")

        if (not args.train_distributed) or (args.train_distributed and args.rank == 0):
            wandb.finish()

        # if args.train_distributed:
        #     cleanup()
