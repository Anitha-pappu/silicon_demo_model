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
from pprint import pformat

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from preprocessing.tags_config.tags_config_loader import (
    BaseConfigLoader,
    ConfigLoadError,
)
from preprocessing.tags_config.tags_config_schemas import BaseTagParameters

TTagParameters = tp.TypeVar('TTagParameters', bound=BaseTagParameters)


class TagsConfig(tp.Generic[TTagParameters], tp.Mapping[str, TTagParameters]):
    """
    A mapping from tag names to their respective configuration parameters.

    Attributes:
        _config: A list of model instances representing the configuration parameters.
        _tag_to_config: A dictionary mapping tag names to the configuration parameters.
        _model_schema: The Pydantic model class used for validation of the parameters.

    Usage example:
        # Define a Pydantic model for tag parameters
        class TagMeta(BaseModel):
            tag_name: str
            description: str
            value: float

        # Initialize TagsConfig with a list of TagMeta instances
        tags_meta = [
            TagMeta(tag_name='Temperature', description='Temp sensor', value=23.5),
            TagMeta(tag_name='Pressure', description='Pressure sensor', value=1.8)
        ]
        tags_config = TagsConfig(parameters=tags_meta, model_schema=TagMeta)

        # Access tag configuration by tag_name
        temperature_config = tags_config['Temperature']
        print(temperature_config.description)  # Output: Temperature sensor

        # Load configuration from a CSV file
        csv_loader = CSVConfigLoader(delimiter=';')
        tags_config_loaded = TagsConfig.load_config(
            path='path/to/config.csv',
            loader=csv_loader,
            model=TagMeta
        )
    """

    def __init__(
        self,
        tags_parameters: tp.Iterable[TTagParameters],
        model_schema: tp.Type[TTagParameters],
    ) -> None:
        """
        Initialize the TagsConfig with an iterable of parameters and a model schema.

        Args:
            tags_parameters: An iterable of model instances representing
            the configuration parameters.
            model_schema: Pydantic model class used for validation of the parameters.
        """
        self._config = TypeAdapter(
            list[TTagParameters],
        ).validate_python(tags_parameters)
        self._tag_to_config: dict[str, TTagParameters] = {
            params.tag_name: params for params in self._config
        }
        self._model_schema = model_schema

    def __getitem__(self, tag_name: str) -> TTagParameters:
        """
        Get the configuration parameters for the given tag name.

        Args:
            tag_name: The name of the tag to retrieve parameters for.

        Returns:
            The configuration parameters for the given tag name.

        Raises:
            TypeError: If the tag_name is not a string.
        """
        if not isinstance(tag_name, str):
            raise TypeError(f"Unknown key type: {type(tag_name).__name__}")
        return self._tag_to_config[tag_name]

    def __iter__(self) -> tp.Iterator[str]:
        """Iterate over the tag names in the configuration."""
        return iter(self._tag_to_config)

    def __len__(self) -> int:
        """Get the number of tags in the configuration."""
        return len(self._tag_to_config)

    def __repr__(self) -> str:
        """Return a string representation of the TagsConfig."""
        class_name = self.__class__.__name__
        model_name = self._model_schema.__name__
        keys = pformat(list(self._tag_to_config.keys()), indent=8)
        fields = pformat(self._model_schema.model_fields.keys(), indent=8)
        return (
            f"{class_name}[{model_name}](\n"
            f"    keys={keys},\n"
            f"    model_fields={fields}\n"
            f")"
        )

    def to_df(self) -> pd.DataFrame:
        """
        Convert all the models in TagsConfig into a single pandas DataFrame.

        Returns:
            A pandas DataFrame containing all the configuration parameters.
        """
        data_dicts = [model.dict() for model in self._config]
        return pd.DataFrame(data_dicts)

    @classmethod
    def load_config(
        cls,
        path: str,
        loader: BaseConfigLoader,
        model: tp.Type[TTagParameters],
        rename_mapping: dict[str, str] | None = None,
    ) -> "TagsConfig[TTagParameters]":
        """
        Load configuration from a file using the specified loader and model schema.

        Args:
            path: The path to the configuration file.
            loader: An BaseConfigLoader instance to use for loading the configuration.
            model: The Pydantic model class to use for validation of the configuration.
            rename_mapping: An optional dictionary to rename keys in the  configuration.

        Returns:
            An instance of TagsConfig with the loaded configuration.

        Raises:
            ConfigLoadException: loading or validating error.
        """
        try:  # noqa: WPS229
            dict_config = loader.load(path)

            if rename_mapping:
                dict_config = [
                    {
                        rename_mapping.get(key, key): config_entry
                        for key, config_entry in entry.items()
                    }
                    for entry in dict_config
                ]

            config = TypeAdapter(
                list[model],  # type: ignore
            ).validate_python(dict_config)
        except ConfigLoadError as exc:
            raise ConfigLoadError(
                f"Failed to load configuration due to loading error: {exc}",
            )
        except ValidationError as exc:
            raise ConfigLoadError(
                f"Failed to load configuration due to validation error: {exc}",
            )

        return cls(config, model)
