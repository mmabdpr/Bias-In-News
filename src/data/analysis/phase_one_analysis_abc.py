import logging
import os
from abc import ABC, abstractmethod
from typing import *


class PhaseOneAnalysisABC(ABC):

    def check_for_required_files(self):
        logging.info("checking for required files")

        for file_path in self.required_files:
            if not os.path.isfile(file_path):
                logging.error(f"required file {file_path} was not found")
                raise FileNotFoundError(f'{file_path} not found')

        logging.info("required files check passed")

    @property
    @abstractmethod
    def required_files(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def report_name(self) -> str:
        pass

    @abstractmethod
    def generate_tables(self):
        pass

    @abstractmethod
    def generate_graphs(self):
        pass
