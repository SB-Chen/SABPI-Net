import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=f'Training for classification!')
    parser.add_argument(
        "--model_name", required=True, type=str, help="Train modeling name",
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed for initialization')
    parser.add_argument('--dataset_name', default='None', type=str, help='dataset name for train')
    parser.add_argument('--exp_str', default='0', type=str, help='additional info to indicate which experiment it is')
    parser.add_argument(
        "--device", required=True, type=str, help="gpu number, for example:'cuda:0'"
    )
    parser.add_argument(
        "--epoch", "-epochs", required=True, type=int, help="train epoch number"
    )  #
    parser.add_argument(
        "--batch_size", required=True, type=int, help="batch_size"
    )
    parser.add_argument(
        "--train_distributed", default=False, type=bool, help="whether to train distributed with multiple GPU"
    )
    parser.add_argument(
        "--k_folds", nargs='+', type=int, help="cross validation fold numbers list!"
    )  # required=True,
    parser.add_argument(
        "--syncBN", default=True, type=bool, help="SyncBatchNorm only makes sense when training networks that have BN structures."
    )
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--resume", type=str, default=None, help="modeling resume")
    parser.add_argument(
        "--loss_type", type=str, default='crossEntropyLoss', help="loss function"
    )
    parser.add_argument(
        "--model_optimizer", type=str, default='SGD', help="model_optimizer"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default='step',
        help="lr_scheduler", choices=['step', 'cos_lr_epoch', 'cos_lr_step']
    )
    parser.add_argument(
        "--base_learning_rate", type=float, default='1e-3', help="base_learning_rate"
    )
    parser.add_argument(
        "--pre_trained", type=bool, default=False, help="pre_trained"
    )
    parser.add_argument(
        "--distill_epoch", type=int, default=0, help="epoch to start distillation"
    )
    parser.add_argument("--temperature", default=1, type=float,
                        help='use in kd loss for dual_branch_trainingUtil_classification')
    parser.add_argument("--alpha", default=2, type=float,
                        help='use in kd loss for dual_branch_trainingUtil_classification')
    parser.add_argument("--feature_vec_dim", default=1024, type=int,
                        help='use in kd loss for dual_branch_trainingUtil_classification')
    parser.add_argument(
        "--train_record_save_root", type=str,
        default='',
        help="train record save path"
    )
  
    parser.add_argument(
        "--save_model_num", type=int, default=0, help="Number of saved models"
    )
    parser.add_argument(
        "--save_model_fre_epoch", type=float, default=10, help="frequency to save modeling"
    )
    parser.add_argument(
        "--DeiT_teacher", type=str,
        default=None,
        help="teacher modeling of DeiT!"
    )
    parser.add_argument('--evaluation_Frequency', default=1, type=int,
                        help='every evaluationFrq epoch to do a evaluation')
    parser.add_argument('--warmup_epochs', default=1, type=int,
                        help='Every warmup_epochs epochs, multiply the learning rate by gamma to perform decay.')
    parser.add_argument('--end_lr_cos', default=0.0, type=float,
                        help='The minimum learning rate in the CosineAnnealingLR cosine annealing learning rate update strategy')
    parser.add_argument('--num_worker', default=1, type=int, help='data loader worker number.')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4) for SGD',
                        dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum in SGD')

    parser.add_argument('--train_rule', default='None', type=str,
                        help='data sampling strategy for train loader, or use in loss function for classes weights')
    # use in Long-Tailed
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--rho_steps', metavar='N', type=float, nargs='+', help='rho values at various steps')
    parser.add_argument('--min_rho', type=float, default=0.05)
    parser.add_argument('--max_rho', type=float, default=0.8)
    parser.add_argument('--rho_schedule', default='none',
                        choices=('none', 'linear', 'step'))
    parser.add_argument('--margin', default=0.5, type=float, metavar='M',
                        help='Margin value for LDAM')
    print(parser.description)
    arg = parser.parse_args()
    return arg
