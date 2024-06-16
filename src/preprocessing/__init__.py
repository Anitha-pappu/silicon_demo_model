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
"""
This is a boilerplate pipeline
"""
from .anomaly_detector import (  # noqa: F401
    AnomalyDetector,
    MissingValuesDetector,
    RangeDetector,
    create_detectors_dict,
    detect_data_anomaly,
)
from .cleaning import (  # noqa: F401
    apply_outlier_remove_rule,
    apply_type,
    convert_bool,
    deduplicate_pandas,
    enforce_custom_schema,
    enforce_schema,
    remove_null_columns,
    remove_outlier,
    replace_inf_values,
    series_convert_bool,
    unify_timestamp_col_name,
)
from .configs import get_tag_config
from .imputing import (  # noqa: F401
    ModelBasedImputer,
    interpolate_cols,
    transform_numeric_imputer,
)
from .on_off_logic import set_off_equipment_to_zero  # noqa: F401
from .resampling import resample_data  # noqa: F401
from .tags_config import (
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
from .timezones import round_timestamps  # noqa: F401
from .utils import (  # noqa: F401
    calculate_tag_range,
    count_outlier,
    count_outside_threshold,
    create_summary_table,
    get_tag_range,
    preprocessing_output_summary,
    rename_tags,
    update_tag_range,
)

__version__ = "0.19.0"
