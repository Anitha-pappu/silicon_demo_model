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

from kedro.io.core import Version
from kedro_datasets.pandas import CSVDataset

from preprocessing import TagsConfig, TTagParameters, get_tag_config

ConfigDict = tp.Optional[tp.Dict[str, tp.Any]]


class TagsConfigCSVDataset(CSVDataset):

    def __init__(
        self,
        filepath: str,
        load_args: ConfigDict = None,
        save_args: ConfigDict = None,
        version: tp.Optional[Version] = None,
        credentials: ConfigDict = None,
        fs_args: ConfigDict = None,
    ) -> None:
        self._tc_load_args = {
            load_arg: load_args.pop(load_arg)
            for load_arg in frozenset(("delimiter", "parameters_schema"))
            if load_arg in load_args
        } if load_args else {}

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
        )

    def _load(self) -> TagsConfig[TTagParameters]:
        return get_tag_config(
            str(self._filepath),
            "csv",
            **self._tc_load_args,
        )

    def _save(self, config: TagsConfig[TTagParameters]) -> None:
        df = config.to_df()
        super()._save(df)
