from abc import ABC, abstractmethod
import pandas as pd

class DataFetcher(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, **kwargs) -> pd.DataFrame:
        """
        Fetch OHLCV data for `symbol` at `interval`, return a DataFrame
        with columns ['open','high','low','close','volume'] indexed by datetime.
        """
        pass