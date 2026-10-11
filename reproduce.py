"""Verify the retained educational solver against saved portfolio output."""
from pathlib import Path
import json
import csv
import numpy as np
from heat_model import Parameters, solve
root=Path(__file__).parent
saved=json.loads((root/'heat-results.json').read_text(encoding='utf-8'))
p=Parameters(**saved['parameters'])
result=solve(p,snapshot_times=())
residual=float(np.max(np.abs(result.temperature[0,:]-saved['axis_temperature'])))
if residual > 1e-9:
    raise AssertionError(f'Saved axis mismatch: {residual}')
with (root/'reproduced-field.csv').open('w', encoding='utf-8', newline='') as f:
    w=csv.writer(f);w.writerow(['radius','theta','temperature'])
    for i,r in enumerate(result.radius):
        for j,theta in enumerate(result.theta):
            w.writerow([r,theta,result.temperature[i,j]])
print(f'Grid={result.temperature.shape}, axis residual={residual:.3g}')
print('Saved temporal error:',saved['history'][-1]['max_error'])
