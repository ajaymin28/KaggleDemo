import argparse

def getSpampinatoArgs():
    parser = argparse.ArgumentParser('EEG Dom Adv training')

    # Data args
    parser.add_argument('--subject',
                        type=int,
                        default=1,
                        help='Subject Data to train')
    parser.add_argument('--dataset',
                        type=str,
                        default="./data/spampinato/eeg/eeg_signals_raw_with_mean_std.pth",
                        help='Dataset to train')
    parser.add_argument('--dataset_split',
                        type=str,
                        default="./data/spampinato/eeg/block_splits_by_image_all.pth",
                        help='Dataset split')
    parser.add_argument('--imagenet_root',
                        type=str,
                        default="./data/spampinato/images/imageNet_images",
                        help='Dataset split')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    # Train Args
    parser.add_argument('--learning_rate',
                        type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=512,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--custom_model_weights',
                        type=str,
                        default="",
                        help='custom model weights')
    # misc
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--query_dataset',
                    type=str,
                    default="EEG",
                    help='EEG,caltech101')
    parser.add_argument('--search_gallary',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--query_gallary',
                        type=str,
                        default="domainnet",
                        help='dataset in which images will be searched')
    parser.add_argument('--domainnet_subtype',
                        type=str,
                        default="clipart",
                        help='Sub type of Domainnet dataset')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--class_to_search',
                        type=str,
                        default="",
                        help='dataset class to search ')
    parser.add_argument('--imagenet_label_name',
                        type=str,
                        default="",
                        help='imagenet label class name')
    

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


def getThingsArgs():
    parser = argparse.ArgumentParser('EEG Dom Adv training for ThingsDataset')
    
    # Data args
    parser.add_argument('--subject',
                        type=int,
                        default=1,
                        help='Subject Data to train')
    parser.add_argument('--dataset',
                        type=str,
                        default="./data/spampinato/eeg/eeg_signals_raw_with_mean_std.pth",
                        help='Dataset to train')
    parser.add_argument('--dataset_split',
                        type=str,
                        default="./data/spampinato/eeg/block_splits_by_image_all.pth",
                        help='Dataset split')
    parser.add_argument('--imagenet_root',
                        type=str,
                        default="./data/spampinato/images/imageNet_images",
                        help='Dataset split')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    # Train Args
    parser.add_argument('--learning_rate',
                        type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=512,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--custom_model_weights',
                        type=str,
                        default="",
                        help='custom model weights')
    # misc
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed