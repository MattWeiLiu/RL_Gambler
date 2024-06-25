from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import math

window_size = 15

df = pd.read_csv("../data/luxury_dice_game.csv")
df = df.set_index("roundNumber")
df = df.sort_index()
df = pd.get_dummies(df, columns=["Open"], dtype=int)
df["NumBet"] = df["NumBet"].apply(lambda x: math.log(x, 100))

record = []
time_code = []
y = []
for i in tqdm(range(window_size, df.shape[0])):
    tmp = df.iloc[i - window_size:i, :]
    record.append(np.concatenate([tmp.iloc[:, 3:].values]))
    time_code.append(df.iloc[i, 1:3].values.flatten())
    label = df.iloc[i, 3:16].values
    y.append(label)

np.savez("luxury_dice_v2.npz", record=np.array(record), time_code=np.array(time_code), y=np.array(y))
