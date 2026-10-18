"""Rebuild exact numerical selections for reported statistical tables.

This replication-only version omits publication prose and TeX authoring.
"""
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parent
contrasts = pd.read_csv(P / 'paired_contrasts.csv')
first = contrasts[contrasts.contrast_id.isin([
    'broadcast_minus_ring', 'withdrawn_minus_retained'])]
geometry = contrasts[(contrasts.contrast_id.str.startswith('expanded')) &
                     (contrasts.metric == 'optimal_fraction')]
compact = geometry[(geometry.reference == 'ring') &
                   (geometry.intervention == 'ws0.1')]
for name in ['Main_Table1.csv', 'Table_1_Source_Data.csv']:
    first.to_csv(P / name, index=False)
geometry.to_csv(P / 'Supplement_Table_Geometry.csv', index=False)
for name in ['Main_Table2.csv', 'Table_2_Source_Data.csv']:
    compact.to_csv(P / name, index=False)
print('Wrote five numerical table-selection CSV files.')
