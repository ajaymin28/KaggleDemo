import os
import numpy as np
from torch.utils.data import Dataset
import gc

import sys
sys.path.append("D:\\2025\\Projects\\KaggleDemo")

from utils.common import TrainConfig



class EEGThingsDataset(Dataset):
    def __init__(
        self, 
        args: TrainConfig, 
        nsubs=[1], 
        session_ids=[0], 
        subset="train"
    ):
        assert subset in {"train", "val", "test"}
        assert isinstance(nsubs, list) and len(nsubs) > 0
        assert isinstance(session_ids, list) and len(session_ids) > 0

        self.nsubs = nsubs
        self.session_ids = session_ids
        self.subset = subset
        self.args = args

        # Load global image metadata (shared across subjects)
        img_meta_dir = os.path.join(args.global_config.DATA_BASE_DIR, "Data", "Things-EEG2", 'Image_set')
        img_metadata = np.load(os.path.join(img_meta_dir, 'image_metadata.npy'), allow_pickle=True).item()
        data_key = "train" if subset in {"train", "val"} else "test"
        self.img_file_names = img_metadata[f'{data_key}_img_files']
        self.img_concepts = img_metadata[f'{data_key}_img_concepts']

        # Load image features (same for all subjects, but we'll store a reference per subject for consistency)
        if self.subset=="train" or self.subset=="val":
            img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_training.npy')
        else:
            img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_test.npy')

        subject_img_features_all = np.load(img_feature_path, allow_pickle=True)
        subject_img_features_all = np.squeeze(subject_img_features_all) # (num_total_samples, feature_dim)

        # Build class mappings
        self.class_to_id = {c: i for i, c in enumerate(sorted(set(self.img_concepts)))}

        # For all subjects, load EEG data for selected sessions only
        self.samples = []
        for sub_idx, sub_id in enumerate(nsubs):
            eegfilekey = "training" if data_key == "train" else data_key
            eeg_path = os.path.join(args.global_config.EEG_DATA_PATH, f"sub-{sub_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
            eeg_dict = np.load(eeg_path, allow_pickle=True)
            eeg_data = eeg_dict['preprocessed_eeg_data']  # shape: (samples, sessions, channels, time)
            del eeg_dict  # Immediately free up the dict
            gc.collect()

            # Only select specified session(s)
            for i in range(eeg_data.shape[0]):
                selected_sessions = eeg_data[i, self.session_ids, :, :]  # (len(session_ids), channels, time) or (channels, time) if one session
                # If only one session, squeeze dimension

                # img_file = self.img_file_names[i]
                class_id = self.class_to_id[self.img_concepts[i]]
                image_features = subject_img_features_all[i]

                if selected_sessions.shape[0] == 1:
                    # Only one session, use as is
                    if not self.args.keep_dim_after_mean:
                        session_data = selected_sessions[0]
                    else:
                        session_data = selected_sessions
                    self.samples.append((session_data, class_id, image_features, sub_idx))

                elif selected_sessions.shape[0] > 1:
                    if self.args.mean_eeg_data:
                        # Average across sessions
                        session_data = np.mean(selected_sessions, axis=1, keepdims=self.args.keep_dim_after_mean)
                        self.samples.append((session_data, class_id, image_features, sub_idx))
                    else:
                        # Each session becomes a separate sample
                        for session_data in selected_sessions:
                            self.samples.append((session_data, class_id, image_features, sub_idx))

                # multi_sessions_samples = False
                # if selected_sessions.shape[0] == 1:
                #     selected_sessions = selected_sessions[0]  # (channels, time)
                # elif selected_sessions.shape[0]>1:
                #     if self.args.mean_eeg_data:
                #         selected_sessions = np.mean(selected_sessions,axis=1,keepdims=self.args.keep_dim_after_mean)
                #     else:
                #         multi_sessions_samples = True
                #         # use each session as a sample instead of averaging
                #         for session_i in range(selected_sessions.shape[0]):
                #             selected_sessions = selected_sessions[session_i]
                #             # eeg, label,image,subject, image_features
                #             self.samples.append((
                #                 selected_sessions,     # eeg_data (selected session(s))
                #                 class_id,              # image_class_id
                #                 image_features,        # CLIP image features
                #                 sub_idx,               # zero-indexed subject id
                #             ))

                # if not multi_sessions_samples:
                #     self.samples.append((
                #         selected_sessions,     # eeg_data (selected session(s))
                #         class_id,              # image_class_id
                #         image_features,        # CLIP image features
                #         sub_idx,               # zero-indexed subject id
                #     ))

            
            # Free up eeg_data array as soon as possible
            del eeg_data
            gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    @staticmethod
    def get_eeg_data(args, sessions=[0]):
        """
        Things EEG data have upto 80 sessions for test data (idx 0 to 79)
        Note: Sessions will be averaged

        Use args.nSub for selecting subject(0-9)
        """
        test_data = []
        test_label = np.arange(200)
        test_data = np.load(args.eeg_data_path + '\\sub-' + format(args.nSub, '02') + '\\preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data'][:,sessions,:,:]
        test_data = np.mean(test_data, axis=1, keepdims=False)
        # test_data = np.expand_dims(test_data, axis=1)
        return test_data, test_label
    
if __name__=="__main__":
    

    args = TrainConfig()

    train_ds = EEGThingsDataset(args=args,nsubs=[1], session_ids=[5,6],subset="val")


