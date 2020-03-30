import sys
import pandas as pd
import numpy as np

ROUND_SECONDS = True

file_live = sys.argv[1]
file_pred = sys.argv[2]

data_live = pd.read_csv(file_live, parse_dates=['timestamp'])
data_live.set_index('timestamp', inplace=True)
data_live.rename(columns={'value': 'live'}, inplace=True)

if ROUND_SECONDS:
    data_live.index = data_live.index.round('T')

data_pred = pd.read_csv(file_pred, parse_dates=['timestamp'])
data_pred.set_index('timestamp', inplace=True)
data_pred.rename(columns={'value': 'pred'}, inplace=True)
if ROUND_SECONDS:
    data_pred.index = data_pred.index.round('T')

result = data_live.join(data_pred, how='outer').sort_index(ascending=False)
result['match'] = np.round(result['live'].values) == np.round(result['pred'].values)
print(result[:10])