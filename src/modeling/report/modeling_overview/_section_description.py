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


MODEL_INTRODUCTION_DESCRIPTION = """
    This section shows the output of the repr method of the model,
    offering a concise summary of its structure and parameters.
    It includes information on model type, hyperparameters, and key attributes,
    which can be used to initialize the model with the same configuration.
    """

MODEL_DEFINITION_DESCRIPTION = """
    This section shows the string representation of the ``ModelBase`` model,
    offering a concise summary of its structure and parameters.
    It includes information on model type, hyperparameters, and key attributes,
    which can be used to initialize the model with the same configuration.
    """

MODEL_FACTORY_SECTION_HEADER = """
    This section shows the string representation of the ``ModelFactoryBase``.
    It can be used to initialize the equivalent instance of ``ModelFactoryBase``
    as provided in the report.
    """

MODEL_TUNER_SECTION_HEADER = """
    This section shows the string representation of the ``ModelTunerBase``
    offering a concise summary of hyperparameters configuration used in
    hyperparameters tuning phase. It can be used to initialize the equivalent
    instance of ``ModelTunerBase`` as provided in the report.
    """

VISUAL_REPRESENTATION_DESCRIPTION = """
    This section displays the evolution of the model's target variable over
    time for both the
    training and test periods, while also identifying intervals with missing data.
    """

CONSECUTIVE_PERIODS_DESCRIPTION = """
    This section displays the the consecutive periods from the training and
    testing data sets demonstrating the continuous periods from both data sets.
    """

FEATURE_CORRELATION_DESCRIPTION = """
    This section presents a correlation plot for
    the columns in the provided data sets. Usage of highly correlated features
    in the model can potentially decrease the performance.
    """

TRAIN_TEST_COMPARISON_DESCRIPTION = """
    This section showcases charts comparing feature distributions in
    the training and test sets.
    It aids in detecting potential data drift or domain shift over time,
    enabling better understanding of how the model might perform
    in evolving real-world conditions.
    """

MODEL_ACTUAL_VS_PREDICTED_DESCRIPTION = """
    This section compares the predicted timeline against the actual model target.
    Its purpose is to identify consistent patterns of over- or
    under-prediction by the model.
    """

METRICS_DESCRIPTION = """
    This section showcases the performance metrics of the model and
    showcases performance using multiple metrics to track
    general model quality in general and potential overfitting.
    """

BASELINES_DESCRIPTION = """
    This section compares the performance of the model against multiple baselines
    (simple models that are build using simple rules). Examples:
    1) Moving Average Model (30D) predicts current values as
    an average target values over the last 30 days.
    2) Autoregressive Model (AR1) predicts current
    values using the latest known target values by the moment on prediction.

    This comparison helps understanding if models
    are able to capture any sophisticated features/target dependencies.
    """

RESIDUAL_ANALYSIS_DESCRIPTION = """
    Residual analysis examines the differences between observed and predicted values,
    known as residuals. This section explores patterns within
    residuals to assess model performance.
    It helps identify if the model systematically under- or over-predicts.
    """

FEATURE_IMPORTANCE_TABLE_DESCRIPTION = """
    Feature importance section presents the feature importance table.
    It computes importance using the model's built-in methods and SHAP importance.
    SHAP importance is derived as the average absolute value of SHAP values for the
    feature in the training set.
    """

SHAP_SUMMARY_DESCRIPTION = """
    This section displays the SHAP summary plot. The summary plot
    combines feature importance with feature effects. Each point
    on the summary plot is a Shapley value for a feature and an
    instance. The position on the y-axis is determined by the
     feature and on the x-axis by the Shapley value. The color
    represents the value of the feature from low to high. Overlapping
    points are jittered in y-axis direction, so we get a sense of
    the distribution of the Shapley values per feature.
    """

SHAP_DEPENDENCY_PLOT_DESCRIPTION = """
    SHAP dependence plot is a visualization tool that helps understand the relationship
    between a feature and the model’s prediction.
    It allows you to see how the relationship between the feature and the prediction
    changes as the feature value changes.
    In a SHAP dependence scatter plot, the feature of
    interest is represented along the horizontal axis,
    while the corresponding SHAP values are plotted on the vertical axis.
    Each data point on the scatter plot represents an instance from the dataset,
    with the feature’s value and the corresponding SHAP
    value associated with that instance.
    """

PARTIAL_DEPENDENCY_PLOTS_DESCRIPTION = """
    Partial dependency plots illustrate the relationship between a feature and
    the target variable while accounting for the influence of other features.
    This section visually depicts how the target variable changes as a
    single feature varies, providing insights into its individual
    impact on predictions within the context of the entire feature set.
    """
