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

from preprocessing.tags_config import (  # noqa: WPS235
    CSVConfigLoader,
    TagImputationParameters,
    TagMetaParameters,
    TagOnOffDependencyParameters,
    TagOutliersParameters,
    TagRawParameters,
    TagResampleParameters,
    TagsConfig,
    TTagParameters,
    YAMLConfigLoader,
)

SchemaMap: dict[str, tp.Type[TTagParameters]] = {  # type: ignore
    "raw": TagRawParameters,
    "meta": TagMetaParameters,
    "resample": TagResampleParameters,
    "outliers": TagOutliersParameters,
    "on_off": TagOnOffDependencyParameters,
    "impute": TagImputationParameters,
}


def get_tag_config(
    path_to_tag_config: str,
    config_loader: tp.Literal["csv", "yaml"],
    parameters_schema: tp.Type[TTagParameters] | str,
    delimiter: str = ',',
) -> TagsConfig[TTagParameters]:
    """
    Load tag configurations from a file and return a TagsConfig object.

    Args:
        path_to_tag_config (str): Path to the tag configuration file.
        config_loader (Literal["csv", "yaml"]): Type of config loader to use.
        parameters_schema (str): Pydantic schema for the parameters.
        delimiter (str): Delimiter to use in case of a CSV file.

    Returns:
        TagsConfig: Configurations loaded into a TagsConfig object.
    """
    loader_instance: CSVConfigLoader | YAMLConfigLoader
    if config_loader == "csv":
        loader_instance = CSVConfigLoader(delimiter=delimiter)
    elif config_loader == "yaml":
        loader_instance = YAMLConfigLoader()
    else:
        raise ValueError(f"Unsupported config loader type: {config_loader}")

    if isinstance(parameters_schema, str):
        try:
            parameters_schema = SchemaMap[parameters_schema]
        except KeyError as exc:
            available_schemas = ", ".join(SchemaMap.keys())
            raise ValueError(
                f"Invalid schema type '{parameters_schema}'. "
                f"Available schema types : {available_schemas}",
            ) from exc

    return TagsConfig.load_config(
        path=path_to_tag_config,
        loader=loader_instance,
        model=parameters_schema,
    )
