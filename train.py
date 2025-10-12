from util.trainUtil import classification_trainUtil
from os.path import join
from data_loading.load_data import Load_data_with_AIAM_and_LFBuffer as loadDataFromExcelFile
from util.trainUtil import get_parser_for_classification
from util.trainUtil.distributed_utils import init_distributed_mode, cleanup


if __name__ == '__main__':

    args = get_parser_for_classification()
    data_root = ''

    data_excel_file_path = ''
    excel_column_name = ['image', 'label']
    num_classes = 5
    img_size = 224
    other_args = {
        # 'head_class_idx': [0, 1],
        # 'med_class_idx': [2],
        # 'tail_class_idx': [3],
        'wandb_project_name': args.dataset_name,
        # 'resume_record_file_name': ''
    }  


    if args.train_distributed:
        init_distributed_mode(args=args)
    for kth_fold in args.k_folds:
        train_excel_sheet_name = 'train_{}'.format(kth_fold)
        test_excel_sheet_name = 'valid_{}'.format(kth_fold)
        train_main = classification_trainUtil(args=args,
                                              # dataset_name=dataset_name,
                                              img_path=data_root,
                                              num_classes=num_classes,
                                              img_size=img_size,
                                              data_excel_file_path=data_excel_file_path,
                                              train_excel_sheet_name=train_excel_sheet_name,
                                              val_excel_sheet_name=test_excel_sheet_name,
                                              excel_column_name=excel_column_name,
                                              get_data=loadDataFromExcelFile,
                                              cross_validation_fold=kth_fold,
                                              **other_args)
        train_main.running()

    if args.train_distributed:
        cleanup()
