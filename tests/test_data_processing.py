import pandas as pd
import numpy as np

# We only import optimize_memory because handle_missing_values doesn't exist
from src.data_processing import optimize_memory

def test_optimize_memory_function():
    # 1. Create fake data using 64-bit floats
    df = pd.DataFrame({'serum_creatinine': [1.1, 1.9, 2.0]}, dtype='float64')

    # 2. Run the optimization function
    optimized_df = optimize_memory(df)

    # 3. Verify optimize_memory(df) function reduced it to 32-bit
    assert optimized_df['serum_creatinine'].dtype == 'float32', "Memory was not optimized to float32!"
