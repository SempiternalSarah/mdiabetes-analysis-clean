from copy import deepcopy
from torch import load as tload
import pandas as pd
from pathlib import Path

class ReplayDB:
    # helper class to load data from storage

    def __init__(self, storage_path="local_storage/prod", path_pre=""):
        self.path = Path(path_pre+storage_path).resolve()
        self.t = None
        self._week = None
        self.ext = None

    def files(self):
        assert self.t is not None, "must know replay type"
        p = self.path / self.t
        files = list(p.iterdir())
        return files

    def maxweek(self):
        files = self.files()
        for i in range(len(files)):
            files[i] = files[i].parts[-1].split(".")[0].split("_")[-1]
            files[i] = int(files[i])
        return max(files)

    def minweek(self):
        files = self.files()
        for i in range(len(files)):
            files[i] = files[i].parts[-1].split(".")[0].split("_")[-1]
            files[i] = int(files[i])
        return min(files)

    def week_exists(self, _week):
        assert self.t is not None, "must know replay type"
        name = self.makename()
        p = self.path / self.t / name
        return p.exists()

    def replay(self, data_t):
        self.t = data_t
        if self.t in ["responses", "outfiles"]:
            self.ext = "csv"
        else:
            self.ext = "pt"
        return deepcopy(self)

    def week(self, _week):
        self._week = str(_week) 
        return deepcopy(self)

    def makename(self):
        if self.t == "responses":
            return f"participant_responses_week_{self._week}.{self.ext}"
        elif self.t == "outfiles":
            return f"to_participants_week_{self._week}.{self.ext}"
        else:
            return f"{self._week}.{self.ext}"

    def load(self):
        name = self.makename()
        p = self.path / self.t / name
        if self.ext == "pt":
            return tload(p)
        else:
            return pd.read_csv(p)
