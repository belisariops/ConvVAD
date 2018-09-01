import torch
import torchaudio
from pathlib import Path

from torch.utils.data import Dataset


class VADDataset(Dataset):
    def __init__(self, dir="audios", transform=None):
        self.dir = dir
        self.talking_path = "{}/talking".format(dir)
        self.not_talking_path = "{}/not_talking".format(dir)
        self.audios = []
        talking_file_list = Path(self.talking_path).iterdir()
        not_talking_file_list = Path(self.not_talking_path).iterdir()
        for file in talking_file_list:
            self.audios.append([file, torch.tensor(1)])
        for file in not_talking_file_list:
            self.audios.append([file, torch.tensor(0)])
        self.transform = transform

    def __len__(self):
        talking_file_list = Path(self.talking_path).iterdir()
        not_talking_file_list = Path(self.not_talking_path).iterdir()
        count = sum(1 for x in talking_file_list)
        count += sum(1 for x in not_talking_file_list)
        return count

    def __getitem__(self, index):
        element = self.audios[index]
        audio_name = element[0]
        audio, sample_rate = torchaudio.load(str(audio_name))
        audio = audio.view([1, 200, 240])
        file = [audio, element[1]]

        if self.transform:
            file[0] = self.transform(file[0])

        return file
