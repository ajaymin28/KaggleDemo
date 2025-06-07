from torchvision import transforms

mynet_transform = transforms.Compose([
                transforms.transforms.Resize((224,224)),
                transforms.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.transforms.RandomCrop(size=(32, 32)),
                transforms.transforms.GaussianBlur(kernel_size=(3,3)),
                transforms.transforms.ColorJitter(),
                transforms.transforms.RandomAutocontrast(),
                transforms.transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])