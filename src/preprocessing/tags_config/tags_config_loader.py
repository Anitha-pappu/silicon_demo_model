# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
import typing as tp
from abc import ABC, abstractmethod

import pandas as pd
import yaml


class ConfigLoadError(Exception):
    """
    Custom exception for configuration loading errors.

    This exception is raised when a configuration file cannot be loaded properly,
    either due to file not found, parsing errors, or other IO related issues.
    """


class BaseConfigLoader(ABC):
    """
    Abstract base class for configuration loaders.

    This class defines the interface that all config loaders must implement.
    """

    @abstractmethod
    def load(self, path: str) -> list[dict[str, tp.Any]]:
        """
        Load the configuration file into a dictionary.

        Args:
            path: The file system path to the configuration file.
        Returns:
             A list of dictionaries representation of the configuration.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class CSVConfigLoader(BaseConfigLoader):
    """
    Configuration loader for CSV files.

    This class implements the BaseConfigLoader interface for CSV files.
    """

    def __init__(self, delimiter: str = ",") -> None:
        """
        Initialize the CSVConfigLoader with a specific delimiter.

        Args:
            delimiter: The delimiter character used in the CSV file (default is comma).
        """
        self._delimiter = delimiter

    def load(self, path: str) -> list[dict[str, tp.Any]]:
        """
        Load the configuration from a CSV file.

        Args:
            path: The file system path to the CSV configuration file.
        Returns:
            A list of dictionaries, each representing a row in the CSV file.
        Raises:
            ConfigLoadException: If the CSV file cannot be loaded or parsed.
        """
        try:
            config = pd.read_csv(path, delimiter=self._delimiter)
        except (pd.errors.ParserError, FileNotFoundError) as exc:
            raise ConfigLoadError(f"Failed to load CSV config: {exc}")
        return tp.cast(list[dict[str, tp.Any]], config.to_dict(orient='records'))


class YAMLConfigLoader(BaseConfigLoader):
    """
    Configuration loader for YAML files.

    This class implements the BaseConfigLoader interface for YAML files.
    """

    def load(self, path: str) -> list[dict[str, tp.Any]]:
        """
        Load the configuration from a YAML file.

        Args:
            path: The file system path to the YAML configuration file.
        Returns:
            A list of dictionaries, each representing an entry in the YAML file.
        Raises:
            ConfigLoadException: If the YAML file cannot be loaded or parsed.
        """
        try:  # noqa: WPS229
            with open(path, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.SafeLoader)
            return tp.cast(list[dict[str, tp.Any]], config)
        except (yaml.YAMLError, FileNotFoundError) as exc:
            raise ConfigLoadError(f"Failed to load YAML config: {exc}")
