from google.cloud import bigquery
from lib.utils import time2vec
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def process(num):
    sql = f"SELECT * \
    FROM `aiops-338206.tmp.luxury_dice_20231001-20231209` \
    WHERE roundNumber = {num}"
    query_job = client.query(sql)
    df = query_job.to_dataframe()
    round_timestamp = df.timestamp.min()
    cos, sin = time2vec(round_timestamp)

    tmp = df[df["isWin"] == 1]
    if tmp.shape[0]:
        OPEN = tmp["betType"].values[0]
    else:
        OPEN = "leopard"
    num_bet = df.shape[0]
    p = []
    p.append(df[(df["betType"] == "leopard") & (df["point"] == 100)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "leopard") & (df["point"] == 1000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "leopard") & (df["point"] == 10000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "leopard") & (df["point"] == 100000)].shape[0] / df.shape[0])

    p.append(df[(df["betType"] == "small") & (df["point"] == 100)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "small") & (df["point"] == 1000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "small") & (df["point"] == 10000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "small") & (df["point"] == 100000)].shape[0] / df.shape[0])

    p.append(df[(df["betType"] == "large") & (df["point"] == 100)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "large") & (df["point"] == 1000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "large") & (df["point"] == 10000)].shape[0] / df.shape[0])
    p.append(df[(df["betType"] == "large") & (df["point"] == 100000)].shape[0] / df.shape[0])
    res = [num, round_timestamp, sin, cos, OPEN, num_bet] + p
    with open("luxury_dice_game.csv", "a") as f:
        f.write(",".join([str(k) for k in res]) + "\n")


def parallel_process_dataframe(func, round_number, max_workers=None):
    with ThreadPoolExecutor(max_workers) as executor:
        list(tqdm(executor.map(func, round_number), total=len(round_number)))


client = bigquery.Client(project='media17-1119')
columns = ["roundNumber", "Timestamp", "SINE", "COSINE", "Open", "NumBet",
           "Leopard_100", "Leopard_1000", "Leopard_10000", "Leopard_100000",
           "Small_100", "Small_1000", "Small_10000", "Small_100000",
           "Large_100", "Large_1000", "Large_10000", "Large_100000"]

sql = """
SELECT DISTINCT(roundNumber) FROM `aiops-338206.tmp.luxury_dice_20231001-20231209`
"""
query_job = client.query(sql)
df = query_job.to_dataframe()
round_number = sorted(df["roundNumber"].values.tolist())
d = pd.read_csv("luxury_dice_game.csv")
round_number = list(set(round_number) - set(d["roundNumber"].to_list()))

# with open("luxury_dice_game.csv", "w") as f:
#     f.write(",".join(columns)+"\n")
parallel_process_dataframe(process, round_number, max_workers=7)
