from torch.utils.data import Dataset as DSRC
import numpy as np
import os
import cv2
import torch


def encode_classes(input, num_classes):
    output = torch.zeros((input.size(0), num_classes),
                         dtype=torch.float32,
                         device=input.device)
    for i in range(input.size(0)):
        output[i][input[i]] = 1
    return output


class Dataset(DSRC):
    def __init__(self, path, extensions=['png']) -> None:
        super().__init__()
        assert os.path.isdir(path), f'Path not provided or incorrect: {path}'
        self.path = path
        self.classes = [
            int(d) for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        counter = 0
        self.index = {}
        for c in self.classes:
            files = os.listdir(os.path.join(self.path, str(c)))
            for file in files:
                for ext in extensions:
                    if file.endswith(ext):
                        self.index[counter] = [file, c]
                        counter += 1

    def __getitem__(self, index):
        fname, c = self.index[index]
        path = os.path.join(self.path, str(c), fname)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img.transpose(2, 0, 1), self.encode_class(c)

    def __len__(self):
        return len(self.index)

    def encode_class(self, c):
        output = np.zeros(len(self.classes), dtype=np.float32)
        output[c] = 1
        return output