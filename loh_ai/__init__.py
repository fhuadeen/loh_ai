import abc
import os
import sys
from typing import List, Any, Union

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# from loh_utils.databases import Database

class LoHAI(abc.ABC):
    """Base LoH class. An abstract class that can be inherited from"""
    def __init__(self):
        pass


class BaseVectorDB(LoHAI):
    def create_embeddings_db(
        self,
        db_type: str,
        docs: List[Any],
        embeddings_obj: Any,
    ):
        pass


class LoHModel(LoHAI):
    def __init__(self, model_name: Union[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model_name

    def get_model(self):
        return self.model

    # def set_model(self, model_name):
    #     self.model_name = model_name


class AIChainer(LoHAI):
    ...
