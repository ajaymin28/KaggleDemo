import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DomainNetDataset(Dataset):
    def __init__(
        self,
        root_dir,        # e.g. "DomainNet/"
        domains,         # e.g. ["clipart", "sketch"]
        split='train',   # "train" or "test"
        classes=None,    # list of class names to include, or None for all
        transform=None,
    ):
        self.samples = []
        self.transform = transform or transforms.ToTensor()
        self.class_names_set = set()  # Collect all class names

        # Load samples, collect class names
        for domain in domains:
            txt_file = os.path.join(root_dir, f"{domain}_{split}.txt")
            if not os.path.exists(txt_file):
                raise FileNotFoundError(f"Split file {txt_file} not found!")

            with open(txt_file, 'r') as f:
                for line in f:
                    rel_path, _ = line.strip().split()  # Ignore index in txt!
                    class_name = rel_path.split('/')[1]
                    if classes is not None and class_name not in classes:
                        continue
                    img_path = os.path.join(root_dir, rel_path)
                    self.samples.append({
                        "img_path": img_path,
                        "domain": domain,
                        "class_name": class_name
                    })
                    self.class_names_set.add(class_name)

        # Sort and build class-to-idx mapping
        if classes is not None:
            self.class_names = sorted(set(classes) & self.class_names_set)
        else:
            self.class_names = sorted(self.class_names_set)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        # Build domain-to-index
        all_domains = sorted({sample['domain'] for sample in self.samples})
        self.domain_to_idx = {d: i for i, d in enumerate(all_domains)}

        # Assign final label indices
        for sample in self.samples:
            sample['class_idx'] = self.class_to_idx[sample['class_name']]
            sample['domain_idx'] = self.domain_to_idx[sample['domain']]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert('RGB')
        img = self.transform(img)
        return {
            "image": img,
            "label": item["class_idx"],
            "domain": item["domain_idx"],
            "domain_name": item["domain"],
            "class_name": item["class_name"]
        }