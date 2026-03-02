# PyTorch Dataset for loading simulations now saved as .npz files.
# Each item returns:
#   - arr: 1D torch.float32 tensor (preprocessed data)
#   - optical_props: torch.float32 tensor (numeric parameters read from the table row)


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class npzDataset(Dataset):
    def __init__(self, df, npz_dir):
        self.df = df
        self.npz_dir = npz_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.df.iloc[idx, 0]
        optical_props = self.df.iloc[idx, 1:].values.astype(np.float32)
        npz_path = os.path.join(self.npz_dir, f'{file_id.rsplit('.mat')[0]}.npz')
        npz_data = np.load(npz_path)


        arr = np.asarray(npz_data["output"])
        # Preprocessing:
        # - drop first two entries as safe measure against NaNs
        # - log transform with shift to keep argument positive
        # - normalise by max
        arr = np.log(arr[2:] + np.min(arr[2:]) + 1.01)
        arr = arr / np.max(arr)
        arr = torch.tensor(arr, dtype=torch.float32)
        optical_props = torch.tensor(optical_props, dtype=torch.float32)
        return arr, optical_props

# Usage
if __name__ == '__main__':
    import pandas as pd
    tabella_path = '../ProcessedPoisson/legenda.txt'
    npz_dir = os.path.dirname(tabella_path)

    df = pd.read_csv(tabella_path, delim_whitespace=True)
    df.info()
    # for all filenames in the first column check if the file exists and create a dataframe with only the existing files
    def npz_exists(x):
        base = str(x).rsplit(".mat", 1)[0]
        return os.path.exists(os.path.join(npz_dir, f"{base}.npz"))

    df = df[df.iloc[:, 0].apply(npz_exists)]
    df.info()

    dataset = npzDataset(df, npz_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for batch in dataloader:
        print('Batch size:', batch[0].size())
        break