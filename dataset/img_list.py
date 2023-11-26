import PIL
from torch.utils.data import Dataset


class ImageListDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]

        if isinstance(image, PIL.Image.Image):
            img = image
        else:
            img = PIL.Image.open(image)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, index