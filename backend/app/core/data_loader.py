from pathlib import Path
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


from app.core.paths import get_data_folder, get_dataset_path



def load_dataset(filename: str, target_column: str, read_csv_kwargs: Dict[str, Any] = None) -> pd.DataFrame:
    """Load the dataset from backend/data and verify the target column exists."""
    read_csv_kwargs = read_csv_kwargs or {"sep": ",", "encoding": "utf-8"}
    dataset_path = get_dataset_path(filename)

    logger.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path, **read_csv_kwargs)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
        )

    logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])
    return df
