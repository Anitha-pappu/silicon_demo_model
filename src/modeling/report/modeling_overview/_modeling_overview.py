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

import logging
import typing as tp

import pandas as pd

from modeling.api import (
    ContainsEstimator,
    ShapExplanation,
    SupportsEvaluateMetrics,
    SupportsModel,
    SupportsModelFactory,
    SupportsModelTuner,
)
from modeling.report.charts import (
    create_table_train_test_metrics,
    plot_actual_vs_predicted,
    plot_actual_vs_residuals,
    plot_partial_dependency_for_sklearn,
    plot_partial_dependency_plots,
    plot_shap_dependency,
    plot_shap_summary,
)
from modeling.report.charts.benchmark_models import (
    fit_default_benchmark_models,
)
from reporting.config import with_default_pyplot_backend
from reporting.rendering import identifiers
from reporting.rendering.types import (
    TRenderingAgnosticContent,
    TRenderingAgnosticDict,
)

from . import _section_description as section_descriptions  # noqa: WPS450
from ._config_parser import (  # noqa: WPS436
    ModelPerformanceConfig,
    PDPSectionConfig,
    TRawConfig,
    ValidationApproachConfig,
    parse_config,
)
from ._types import Model, SupportsModelAndEvaluateMetrics  # noqa: WPS436
from ._validation_approach import plot_validation_approach  # noqa: WPS436

logger = logging.getLogger(__name__)

_TBaselineModels = tp.Dict[str, SupportsModelAndEvaluateMetrics]


_NO_TITLE_LAYOUT_UPDATE = dict(title_text=None, margin_t=60)

_TABLES_WIDTH = 100
_SHAP_DEPENDENCY_SUBPLOT_WIDTH = 530
_SHAP_DEPENDENCY_SUBPLOT_HEIGHT = 400
_SHAP_DEPENDENCY_SPACING_PER_ROW = 0.52
_SHAP_DEPENDENCY_SPACING_PER_COLUMN = 0.4
_SHAP_SUMMARY_WIDTH = 800
_SKLEARN_PDP_PLOT_WIDTH = 12


@with_default_pyplot_backend  # type: ignore  # mypy issue
def get_modeling_overview(
    model: Model,
    timestamp_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_tuner: tp.Optional[SupportsModelTuner] = None,
    model_factory: tp.Optional[SupportsModelFactory] = None,
    baseline_models: tp.Optional[_TBaselineModels] = None,
    model_performance_config: TRawConfig = None,
    pdp_section_config: TRawConfig = None,
    validation_approach_config: TRawConfig = None,
) -> TRenderingAgnosticDict:
    """
    Generates multi-level dict of figures for the Performance Report.

    There are 5 main sections in this report:
        * Model Introduction (model specification, features, target)
        * Validation Approach (visual of validation schema, features & target split
         comparisons, correlation heatmaps)
        * Model Performance (model metrics and core visuals to access model quality)
        * Residual Analysis (deep dive in potential root causes of low performance)
        * Feature Importance (deep dive into main performance drivers)

    This report can be viewed in the notebook or
    used in report generation to produce standalone report file.

    Args:
        timestamp_column: column name of timestamp
        train_data: data containing train input features
        test_data: data containing test input features
        model: trained model
        model_tuner: tuner used to find optimal hyperparameters for model
        model_factory: model factory used to produce model
        model_performance_config: dict cast to `ModelPerformanceConfig` with attrs:
            performance_table_sort_by: metric used for sorting in model performance
             metrics
            performance_table_sort_order: sorting order
        baseline_models: mapping from names to models used for comparison with `model`;
            typically we want to have:
            * simple reference models here like AR1, previous month average,
            * reference models from previous iterations.
             By default, we provide AR1 model in addition to all baseline models passed.
        pdp_section_config: dict cast to `PDPSectionConfig` with attrs:
            * max_features_to_display: maximum number of subplots to display
            * n_point_in_grid: number of points in grid for pdp
            * grid_calculation_strategy: strategy for grid calculation.
             Supported values: "quantiles", "uniform", "quantiles+uniform"
            * y_axis_range_mode: mode for y-axis range default view.
             Supported values: "average", "all".
            * n_sample_to_calculate_predictions: number of samples to sample
             from data to calculate predictions
            * random_state: used for producing sampling
        validation_approach_config: dict cast to `ValidationApproachConfig` with attrs:
            * sort_feature_comparison_by_shap: sorts train/test feature
                comparison in validation section by shap importance if true,
                leaves `features_in` order otherwise
    Returns:
        Dictionary of model performance figures
    """

    train_data = train_data.copy()
    test_data = test_data.copy()

    prediction_column = "__prediction"
    train_data[prediction_column] = model.predict(train_data)
    test_data[prediction_column] = model.predict(test_data)

    shap_explanation = model.produce_shap_explanation(train_data)

    # todo: add description for each section
    #  in performance maybe use plots numbering
    figs = {
        identifiers.SectionHeader(
            header_text="Model Introduction",
            description=section_descriptions.MODEL_INTRODUCTION_DESCRIPTION,
        ): _get_introduction(model, model_tuner, model_factory),
        identifiers.SectionHeader(
            header_text="Validation Approach",
            description=None,
        ): _get_validation_approach(
            model.target,
            model.features_in,
            model.get_shap_feature_importance_from_explanation(shap_explanation),
            timestamp_column,
            train_data,
            test_data,
            validation_config=parse_config(
                validation_approach_config, ValidationApproachConfig,
            ),
        ),
        identifiers.SectionHeader(
            header_text="Model Performance",
            description=None,
        ): _get_model_performance(
            timestamp_column,
            prediction_column,
            train_data,
            test_data,
            model,
            baseline_models,
            config=parse_config(model_performance_config, ModelPerformanceConfig),
        ),
        identifiers.SectionHeader(
            header_text="Residual Analysis",
            description=section_descriptions.RESIDUAL_ANALYSIS_DESCRIPTION,
        ): _get_residual_analysis(
            timestamp_column,
            prediction_column,
            train_data,
            test_data,
            model,
        ),
        identifiers.SectionHeader(
            header_text="Feature Importance",
            description=None,
        ): _get_feature_importance(
            model,
            train_data,
            shap_explanation,
            pdp_config=parse_config(pdp_section_config, PDPSectionConfig),
        ),
    }
    # assigning to variable name figs
    # for simplicity and readability
    return figs  # noqa: WPS331


def _get_introduction(
    model: SupportsModel,
    model_tuner: tp.Optional[SupportsModelTuner],
    model_factory: tp.Optional[SupportsModelFactory],
) -> TRenderingAgnosticDict:
    introduction = {
        identifiers.SectionHeader(
            header_text="Model Definition",
            description=section_descriptions.MODEL_DEFINITION_DESCRIPTION,
        ): identifiers.Code(code=repr(model), code_formatter="py-black"),
        "Target": identifiers.Code(code=str(model.target)),
        "Features": identifiers.Code(code=str("\n".join(model.features_in))),
    }
    if model_tuner is not None:
        model_tuner_section_header = identifiers.SectionHeader(
            header_text="Model Tuner",
            description=section_descriptions.MODEL_TUNER_SECTION_HEADER,
        )
        introduction[model_tuner_section_header] = identifiers.Code(
            code=repr(model_tuner), code_formatter="py-black",
        )
    if model_factory is not None:
        model_factory_section_header = identifiers.SectionHeader(
            header_text="Model Factory",
            description=section_descriptions.MODEL_FACTORY_SECTION_HEADER,
        )
        introduction[model_factory_section_header] = identifiers.Code(
            code=repr(model_factory), code_formatter="py-black",
        )
    return introduction


def _get_validation_approach(
    target: str,
    features: tp.List[str],
    feature_importance: tp.Dict[str, float],
    timestamp_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    validation_config: ValidationApproachConfig,
) -> TRenderingAgnosticDict:
    """
    Returns: visual representation of validation which contains
        * Visual Representation
        * Consecutive Periods List
        * Correlation heatmaps for train and test datasets
        * Feature on splits comparison
    """
    if validation_config.sort_feature_comparison_by_shap:
        features = sorted(
            features, key=lambda feature: feature_importance[feature], reverse=True,
        )
    return plot_validation_approach(
        target=target,
        features=features,
        timestamp_column=timestamp_column,
        train_data=train_data,
        test_data=test_data,
    )


def _get_model_performance(
    timestamp_column: str,
    prediction_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: Model,
    baseline_models: tp.Optional[_TBaselineModels],
    config: ModelPerformanceConfig,
) -> TRenderingAgnosticDict:
    """ Creates the figures for the overview of the model performance.

    These figures are
    - a table with the performance metrics
    - a graph of actual vs predicted (showing both train and test)
    - (if ``baseline_models`` are provided) a table comparing the performance metrics
     with those of the baseline models
    """
    if baseline_models is None:
        baseline_models = {}
    if config.add_default_baselines:
        benchmark_models = fit_default_benchmark_models(
            target=model.target, data=train_data, timestamp=timestamp_column,
        )
        baseline_models.update(benchmark_models)

    fig_metrics = plot_train_test_metrics(
        train_set_metrics=model.evaluate_metrics(train_data),
        test_set_metrics=model.evaluate_metrics(test_data),
        table_width=_TABLES_WIDTH,
    )
    fig_target_vs_predicted = plot_actual_vs_predicted(
        train_data=train_data,
        test_data=test_data,
        timestamp_column=timestamp_column,
        prediction_column=prediction_column,
        target_column=model.target,
    ).update_layout(_NO_TITLE_LAYOUT_UPDATE)  # removing duplicated title
    model_performance_figures = {
        identifiers.SectionHeader(
            header_text="Metrics",
            description=section_descriptions.METRICS_DESCRIPTION,
        ): fig_metrics,
        identifiers.SectionHeader(
            header_text="Actual Target vs. Predicted",
            description=section_descriptions.MODEL_ACTUAL_VS_PREDICTED_DESCRIPTION,
        ): fig_target_vs_predicted,
    }
    if not baseline_models:
        return model_performance_figures
    fig_baselines = _get_baselines_comparison(
        model,
        baseline_models,
        test_data,
        metric_to_sort_by=config.performance_table_sort_by,
        sort_order=config.performance_table_sort_order,
    )
    model_performance_figures = {
        **model_performance_figures,
        identifiers.SectionHeader(
            header_text="Baselines",
            description=section_descriptions.BASELINES_DESCRIPTION,
        ): fig_baselines,
    }
    return model_performance_figures  # noqa: WPS331  # Naming makes meaning clearer


def _get_baselines_comparison(
    model: SupportsEvaluateMetrics,
    baseline_models: _TBaselineModels,
    test_data: pd.DataFrame,
    metric_to_sort_by: str,
    sort_order: str,
) -> TRenderingAgnosticContent:
    performance_metrics = {
        model_name: model.evaluate_metrics(test_data)
        for model_name, model in baseline_models.items()
    }
    performance_metrics["Current Model"] = model.evaluate_metrics(test_data)
    df_performance_metrics = pd.DataFrame.from_dict(performance_metrics, orient="index")
    return identifiers.Table(
        table=df_performance_metrics,
        width=_TABLES_WIDTH,
        sort_by=[(metric_to_sort_by, sort_order)],
    )


def _get_residual_analysis(
    timestamp_column: str,
    prediction_column: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: Model,
) -> TRenderingAgnosticDict:
    # todo: ? residuals by external category
    return {
        "Actual Target vs. Prediction Residuals": plot_actual_vs_residuals(
            train_data=train_data,
            test_data=test_data,
            timestamp_column=timestamp_column,
            prediction_column=prediction_column,
            target_column=model.target,
        ).update_layout(_NO_TITLE_LAYOUT_UPDATE),  # removing duplicated title
    }


def _get_feature_importance(
    model: Model,
    train_data: pd.DataFrame,
    shap_explanation: ShapExplanation,
    pdp_config: PDPSectionConfig,
) -> TRenderingAgnosticDict:
    default_importance_column = "default_importance"
    shap_importance_column = "shap_importance"
    shap_feature_importance = model.get_shap_feature_importance_from_explanation(
        shap_explanation,
    )
    feature_importance_table = identifiers.Table(
        table=pd.DataFrame(
            {
                default_importance_column: model.get_feature_importance(train_data),
                shap_importance_column: shap_feature_importance,
            },
        ),
        width=_TABLES_WIDTH,
        sort_by=[("shap_importance", "desc")],
        columns_to_color_as_bars=[
            default_importance_column,
            shap_importance_column,
        ],
    )
    shap_summary = (
        plot_shap_summary(
            model.features_in,
            shap_explanation,
            width=_SHAP_SUMMARY_WIDTH,
        )
        # removing duplicated title and increasing margin
        .update_layout(_NO_TITLE_LAYOUT_UPDATE)
    )
    shap_dependency_plot = plot_shap_dependency(
        model.features_in,
        shap_explanation,
        subplot_width=_SHAP_DEPENDENCY_SUBPLOT_WIDTH,
        subplot_height=_SHAP_DEPENDENCY_SUBPLOT_HEIGHT,
        horizontal_spacing_per_row=_SHAP_DEPENDENCY_SPACING_PER_ROW,
        vertical_spacing_per_column=_SHAP_DEPENDENCY_SPACING_PER_COLUMN,
    ).update_layout(_NO_TITLE_LAYOUT_UPDATE)
    partial_dependency_plot = plot_partial_dependency_plots(
        model=model,
        data_for_pdp_grid_calculations=train_data,
        feature_importance=shap_feature_importance,
        max_features_to_display=pdp_config.max_features_to_display,
        n_point_in_grid=pdp_config.n_point_in_grid,
        grid_calculation_strategy=pdp_config.grid_calculation_strategy,
        n_samples_to_calculate_predictions=pdp_config.n_sample_to_calculate_predictions,
        y_axis_tick_values_precision=pdp_config.y_axis_tick_values_precision,
        y_axis_range_mode=pdp_config.y_axis_range_mode,
        random_state=pdp_config.random_state,
    ).update_layout(_NO_TITLE_LAYOUT_UPDATE)
    return {
        identifiers.SectionHeader(
            header_text="Feature Importance Table",
            description=section_descriptions.FEATURE_IMPORTANCE_TABLE_DESCRIPTION,
        ): feature_importance_table,
        identifiers.SectionHeader(
            header_text="SHAP Summary",
            description=section_descriptions.SHAP_SUMMARY_DESCRIPTION,
        ): shap_summary,
        identifiers.SectionHeader(
            header_text="SHAP Dependency Plot",
            description=section_descriptions.SHAP_DEPENDENCY_PLOT_DESCRIPTION,
        ): shap_dependency_plot,
        identifiers.SectionHeader(
            header_text="Partial Dependency Plot",
            description=section_descriptions.PARTIAL_DEPENDENCY_PLOTS_DESCRIPTION,
        ): partial_dependency_plot,
    }


def _add_pdp_section(
    figs: TRenderingAgnosticDict,
    data: pd.DataFrame,
    model: Model,
    include: bool = True,
    plot_individual_lines: bool = True,
    random_state: int = 42,
    n_columns: int = 2,
    per_feature_height: float = 4,
    n_jobs: int = -1,
    drop_missing_values: bool = True,
) -> None:
    logger.warning(
        "This functionality is deprecated. Use "
        "`modeling.report.charts.plot_partial_dependency_plots` instead.",
    )
    if not include:
        return

    if not isinstance(model, ContainsEstimator):
        logger.info("Partial dependency plots are not available for the model used.")
        return

    pdp = plot_partial_dependency_for_sklearn(
        model=model,
        data=data,
        yaxis_title=model.target,
        title=None,
        plot_width=_SKLEARN_PDP_PLOT_WIDTH,
        plot_individual_lines=plot_individual_lines,
        random_state=random_state,
        n_columns=n_columns,
        per_feature_height=per_feature_height,
        n_jobs=n_jobs,
        drop_missing_values=drop_missing_values,
    )
    feature_importance_header = identifiers.SectionHeader(
        header_text="Feature Importance",
        description=None,
    )
    partial_dependency_header = identifiers.SectionHeader(
        header_text="Partial Dependency Plot",
        description=section_descriptions.PARTIAL_DEPENDENCY_PLOTS_DESCRIPTION,
    )
    figs[feature_importance_header][partial_dependency_header] = pdp


def plot_train_test_metrics(
    train_set_metrics: tp.Mapping[str, float],
    test_set_metrics: tp.Mapping[str, float],
    table_width: float,
) -> identifiers.Table:
    split_column = "Split"
    train_and_test_metrics = create_table_train_test_metrics(
        train_set_metrics=train_set_metrics,
        test_set_metrics=test_set_metrics,
        split_column=split_column,
    )
    return identifiers.Table(
        table=train_and_test_metrics,
        width=table_width,
        sort_by=[(split_column, "desc")],
        columns_filters_position=None,
    )
