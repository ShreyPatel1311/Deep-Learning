import time as performance_clock
import os as operating_system
from pathlib import Path as FilePath

matplotlib_cache_directory = FilePath("/tmp/boston-mplconfig")
matplotlib_cache_directory.mkdir(parents=True, exist_ok=True)
operating_system.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_directory))
import matplotlib as plotting_backend
plotting_backend.use("Agg")
import matplotlib.pyplot as plotting_api
import numpy as numerical
import pandas as tabular
import sklearn as scikit_learn_package
import torch as tensor_framework
import torch.nn as neural_modules
from sklearn.base import BaseEstimator as ScikitRegressorBase
from sklearn.base import RegressorMixin as ScikitRegressorMixin
from sklearn.model_selection import KFold as FoldSplitter
from sklearn.model_selection import cross_val_score as evaluate_cross_validation
from sklearn.pipeline import Pipeline as ProcessingPipeline
from sklearn.preprocessing import StandardScaler as FeatureStandardizer

experiment_seed = 7
numerical.random.seed(experiment_seed)
tensor_framework.manual_seed(experiment_seed)

print("torch", tensor_framework.__version__, "| sklearn", scikit_learn_package.__version__,
      "| numpy", numerical.__version__, "| pandas", tabular.__version__)

dataset_columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                   "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_dataframe = tabular.read_csv("housing.csv", sep=r"\s+", header=None,
                                     names=dataset_columns)
print("\nshape:", housing_dataframe.shape)
print(housing_dataframe.describe().T[["mean", "std", "min", "max"]].round(2))

predictor_matrix = housing_dataframe.iloc[:, :13].values.astype(numerical.float32)
house_price_targets = housing_dataframe.iloc[:, 13].values.astype(numerical.float32)


class TorchPipelineRegressor(ScikitRegressorBase, ScikitRegressorMixin):
    """PyTorch regressor that follows the scikit-learn estimator interface."""

    def __init__(self, hidden_layer_sizes=(64, 32), training_epochs=150,
                 mini_batch_size=16, learning_rate=0.001, random_seed=experiment_seed):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.training_epochs = training_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def fit(self, input_features, expected_targets):
        tensor_framework.manual_seed(self.random_seed)
        feature_tensor = tensor_framework.tensor(input_features, dtype=tensor_framework.float32)
        target_tensor = tensor_framework.tensor(expected_targets,
                                                 dtype=tensor_framework.float32).view(-1, 1)
        network_layers = []
        input_feature_count = feature_tensor.shape[1]
        for hidden_layer_width in self.hidden_layer_sizes:
            network_layers += [neural_modules.Linear(input_feature_count, hidden_layer_width),
                               neural_modules.ReLU()]
            input_feature_count = hidden_layer_width
        network_layers.append(neural_modules.Linear(input_feature_count, 1))
        self.trained_network_ = neural_modules.Sequential(*network_layers)
        parameter_optimizer = tensor_framework.optim.Adam(self.trained_network_.parameters(),
                                                           lr=self.learning_rate)
        mean_squared_loss = neural_modules.MSELoss()
        for epoch_index in range(self.training_epochs):
            shuffled_row_indices = tensor_framework.randperm(len(feature_tensor))
            for batch_start_index in range(0, len(feature_tensor), self.mini_batch_size):
                mini_batch_indices = shuffled_row_indices[
                    batch_start_index:batch_start_index + self.mini_batch_size]
                parameter_optimizer.zero_grad()
                mini_batch_loss = mean_squared_loss(
                    self.trained_network_(feature_tensor[mini_batch_indices]),
                    target_tensor[mini_batch_indices])
                mini_batch_loss.backward()
                parameter_optimizer.step()
        return self

    def predict(self, input_features):
        with tensor_framework.no_grad():
            prediction_tensor = self.trained_network_(
                tensor_framework.tensor(input_features, dtype=tensor_framework.float32))
        return prediction_tensor.numpy().ravel()


cross_validation_fold_count = 10
mlp_training_epochs = 150
mlp_mini_batch_size = 16
cross_validation_splitter = FoldSplitter(n_splits=cross_validation_fold_count,
                                         shuffle=True, random_state=experiment_seed)
print(f"\nMLP 13-64-32-1 (relu, adam, mse), k={cross_validation_fold_count}, "
      f"epochs={mlp_training_epochs}, batch={mlp_mini_batch_size}")

evaluation_start_time = performance_clock.time()
unscaled_fold_scores = evaluate_cross_validation(
    TorchPipelineRegressor(training_epochs=mlp_training_epochs,
                           mini_batch_size=mlp_mini_batch_size),
    predictor_matrix, house_price_targets, cv=cross_validation_splitter,
    scoring="neg_mean_squared_error")
print("no scaling:   MSE %.2f (%.2f)  %.1fs" %
      (-unscaled_fold_scores.mean(), unscaled_fold_scores.std(),
       performance_clock.time() - evaluation_start_time))

standardized_regression_pipeline = ProcessingPipeline([
    ("standardize", FeatureStandardizer()),
    ("mlp", TorchPipelineRegressor(training_epochs=mlp_training_epochs,
                                    mini_batch_size=mlp_mini_batch_size)),
])
evaluation_start_time = performance_clock.time()
standardized_fold_scores = evaluate_cross_validation(
    standardized_regression_pipeline, predictor_matrix, house_price_targets,
    cv=cross_validation_splitter, scoring="neg_mean_squared_error")
print("standardized: MSE %.2f (%.2f)  %.1fs" %
      (-standardized_fold_scores.mean(), standardized_fold_scores.std(),
       performance_clock.time() - evaluation_start_time))
print("per fold:", numerical.round(-standardized_fold_scores, 2))


def rectified_linear(linear_values):
    return numerical.maximum(0, linear_values)


def mean_squared_error_value(predicted_values, reference_values):
    return float(numerical.mean((predicted_values.ravel() - reference_values.ravel()) ** 2))


class AdaptiveMomentOptimizer:
    """Small NumPy implementation of the Adam update rule."""

    def __init__(self, parameter_tensors, learning_rate=0.001, first_moment_decay=0.9,
                 second_moment_decay=0.999, stability_epsilon=1e-8):
        self.parameter_tensors = parameter_tensors
        self.learning_rate = learning_rate
        self.first_moment_decay = first_moment_decay
        self.second_moment_decay = second_moment_decay
        self.stability_epsilon = stability_epsilon
        self.first_moment_tensors = [numerical.zeros_like(parameter_tensor)
                                     for parameter_tensor in parameter_tensors]
        self.second_moment_tensors = [numerical.zeros_like(parameter_tensor)
                                      for parameter_tensor in parameter_tensors]
        self.update_count = 0

    def step(self, gradient_tensors):
        self.update_count += 1
        for parameter_tensor, gradient_tensor, first_moment_tensor, second_moment_tensor in zip(
                self.parameter_tensors, gradient_tensors, self.first_moment_tensors,
                self.second_moment_tensors):
            first_moment_tensor[:] = (self.first_moment_decay * first_moment_tensor
                                      + (1 - self.first_moment_decay) * gradient_tensor)
            second_moment_tensor[:] = (self.second_moment_decay * second_moment_tensor
                                       + (1 - self.second_moment_decay) * gradient_tensor ** 2)
            corrected_first_moment = first_moment_tensor / (
                1 - self.first_moment_decay ** self.update_count)
            corrected_second_moment = second_moment_tensor / (
                1 - self.second_moment_decay ** self.update_count)
            parameter_tensor -= self.learning_rate * corrected_first_moment / (
                numerical.sqrt(corrected_second_moment) + self.stability_epsilon)


class SingleHiddenLayerRegressor(ScikitRegressorBase, ScikitRegressorMixin):
    """Fully connected regressor with one ReLU hidden layer."""

    def __init__(self, hidden_unit_count=32, training_epochs=150, mini_batch_size=16,
                 learning_rate=0.001, random_seed=experiment_seed):
        self.hidden_unit_count = hidden_unit_count
        self.training_epochs = training_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def forward(self, input_features):
        hidden_linear_values = input_features @ self.input_to_hidden_weights + self.hidden_bias_vector
        hidden_activations = rectified_linear(hidden_linear_values)
        output_values = hidden_activations @ self.hidden_to_output_weights + self.output_bias_vector
        return hidden_linear_values, hidden_activations, output_values

    def fit(self, input_features, expected_targets):
        random_number_generator = numerical.random.default_rng(self.random_seed)
        sample_count, input_dimension = input_features.shape
        target_matrix = expected_targets.reshape(-1, 1)
        self.input_to_hidden_weights = random_number_generator.normal(
            0, numerical.sqrt(2 / input_dimension), (input_dimension, self.hidden_unit_count))
        self.hidden_bias_vector = numerical.zeros(self.hidden_unit_count)
        self.hidden_to_output_weights = random_number_generator.normal(
            0, numerical.sqrt(2 / self.hidden_unit_count), (self.hidden_unit_count, 1))
        self.output_bias_vector = numerical.zeros(1)
        parameter_optimizer = AdaptiveMomentOptimizer([
            self.input_to_hidden_weights, self.hidden_bias_vector,
            self.hidden_to_output_weights, self.output_bias_vector], self.learning_rate)
        for epoch_index in range(self.training_epochs):
            shuffled_row_indices = random_number_generator.permutation(sample_count)
            for batch_start_index in range(0, sample_count, self.mini_batch_size):
                mini_batch_indices = shuffled_row_indices[
                    batch_start_index:batch_start_index + self.mini_batch_size]
                mini_batch_features = input_features[mini_batch_indices]
                mini_batch_targets = target_matrix[mini_batch_indices]
                hidden_linear_values, hidden_activations, output_values = self.forward(mini_batch_features)
                output_gradients = 2 * (output_values - mini_batch_targets) / len(mini_batch_features)
                hidden_to_output_weight_gradients = hidden_activations.T @ output_gradients
                output_bias_gradients = output_gradients.sum(axis=0)
                hidden_linear_gradients = (output_gradients @ self.hidden_to_output_weights.T) * (
                    hidden_linear_values > 0)
                input_to_hidden_weight_gradients = mini_batch_features.T @ hidden_linear_gradients
                hidden_bias_gradients = hidden_linear_gradients.sum(axis=0)
                parameter_optimizer.step([input_to_hidden_weight_gradients, hidden_bias_gradients,
                                          hidden_to_output_weight_gradients, output_bias_gradients])
        return self

    def predict(self, input_features):
        return self.forward(input_features)[2].ravel()


predictor_matrix_float64 = predictor_matrix.astype(numerical.float64)
house_price_targets_float64 = house_price_targets.astype(numerical.float64)
print("\none hidden layer net (numpy), 10-fold cv, 150 epochs")
print("units   mse     std    time")
hidden_unit_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
mean_fold_errors, fold_error_standard_deviations = [], []
for hidden_unit_count in hidden_unit_candidates:
    single_layer_pipeline = ProcessingPipeline([
        ("standardize", FeatureStandardizer()),
        ("net", SingleHiddenLayerRegressor(hidden_unit_count=hidden_unit_count)),
    ])
    evaluation_start_time = performance_clock.time()
    fold_errors = -evaluate_cross_validation(single_layer_pipeline, predictor_matrix_float64,
                                             house_price_targets_float64, cv=cross_validation_splitter,
                                             scoring="neg_mean_squared_error")
    mean_fold_errors.append(fold_errors.mean())
    fold_error_standard_deviations.append(fold_errors.std())
    print("%5d  %6.2f  %6.2f  %5.1fs" %
          (hidden_unit_count, fold_errors.mean(), fold_errors.std(),
           performance_clock.time() - evaluation_start_time))

plotting_api.figure(figsize=(6, 4))
plotting_api.errorbar(hidden_unit_candidates, mean_fold_errors,
                      yerr=fold_error_standard_deviations, marker="o", capsize=3)
plotting_api.xscale("log", base=2)
plotting_api.axhline(20, color="gray", ls="--", lw=1, label="MSE = 20")
plotting_api.xlabel("hidden units")
plotting_api.ylabel("10-fold CV MSE")
plotting_api.title("One hidden layer, varying width")
plotting_api.legend()
plotting_api.tight_layout()
plotting_api.savefig("hidden_units.png", dpi=150)
plotting_api.close()


class VariableDepthRegressor(ScikitRegressorBase, ScikitRegressorMixin):
    """Fully connected ReLU network with a configurable number of hidden layers."""

    def __init__(self, hidden_layer_sizes=(32, 32), training_epochs=150, mini_batch_size=16,
                 learning_rate=0.001, random_seed=experiment_seed):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.training_epochs = training_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def forward(self, input_features):
        activation_values, preactivation_values = [input_features], []
        current_activation = input_features
        final_layer_index = len(self.weight_matrices) - 1
        for layer_index in range(len(self.weight_matrices)):
            linear_values = current_activation @ self.weight_matrices[layer_index] + self.bias_vectors[layer_index]
            preactivation_values.append(linear_values)
            current_activation = (linear_values if layer_index == final_layer_index
                                  else rectified_linear(linear_values))
            activation_values.append(current_activation)
        return activation_values, preactivation_values

    def backward(self, activation_values, preactivation_values, target_matrix):
        weight_gradients = [None] * len(self.weight_matrices)
        bias_gradients = [None] * len(self.bias_vectors)
        output_gradients = 2 * (activation_values[-1] - target_matrix) / len(target_matrix)
        for layer_index in reversed(range(len(self.weight_matrices))):
            weight_gradients[layer_index] = activation_values[layer_index].T @ output_gradients
            bias_gradients[layer_index] = output_gradients.sum(axis=0)
            if layer_index > 0:
                output_gradients = (output_gradients @ self.weight_matrices[layer_index].T) * (
                    preactivation_values[layer_index - 1] > 0)
        return weight_gradients, bias_gradients

    def fit(self, input_features, expected_targets, validation_features=None,
            validation_targets=None):
        random_number_generator = numerical.random.default_rng(self.random_seed)
        sample_count = len(input_features)
        target_matrix = expected_targets.reshape(-1, 1)
        layer_sizes = [input_features.shape[1]] + list(self.hidden_layer_sizes) + [1]
        self.weight_matrices, self.bias_vectors = [], []
        for input_layer_width, output_layer_width in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weight_matrices.append(random_number_generator.normal(
                0, numerical.sqrt(2 / input_layer_width), (input_layer_width, output_layer_width)))
            self.bias_vectors.append(numerical.zeros(output_layer_width))
        parameter_optimizer = AdaptiveMomentOptimizer(self.weight_matrices + self.bias_vectors,
                                                      self.learning_rate)
        self.training_error_history, self.validation_error_history = [], []
        for epoch_index in range(self.training_epochs):
            shuffled_row_indices = random_number_generator.permutation(sample_count)
            for batch_start_index in range(0, sample_count, self.mini_batch_size):
                mini_batch_indices = shuffled_row_indices[
                    batch_start_index:batch_start_index + self.mini_batch_size]
                activation_values, preactivation_values = self.forward(input_features[mini_batch_indices])
                weight_gradients, bias_gradients = self.backward(
                    activation_values, preactivation_values, target_matrix[mini_batch_indices])
                parameter_optimizer.step(weight_gradients + bias_gradients)
            self.training_error_history.append(mean_squared_error_value(
                self.predict(input_features), target_matrix))
            if validation_features is not None:
                self.validation_error_history.append(mean_squared_error_value(
                    self.predict(validation_features), validation_targets))
        return self

    def predict(self, input_features):
        return self.forward(input_features)[0][-1].ravel()


def first_epoch_at_or_below(loss_history, error_threshold):
    qualifying_epoch_indices = numerical.where(numerical.array(loss_history) <= error_threshold)[0]
    return (qualifying_epoch_indices[0] + 1 if len(qualifying_epoch_indices) > 0
            else numerical.nan)


scaled_predictor_matrix = FeatureStandardizer().fit_transform(predictor_matrix_float64)
single_layer_predictions = SingleHiddenLayerRegressor(hidden_unit_count=16,
                                                      training_epochs=5).fit(
    scaled_predictor_matrix, house_price_targets_float64).predict(scaled_predictor_matrix)
depth_network_predictions = VariableDepthRegressor(hidden_layer_sizes=(16,),
                                                    training_epochs=5).fit(
    scaled_predictor_matrix, house_price_targets_float64).predict(scaled_predictor_matrix)
print("\nVariableDepthRegressor(1 layer) vs SingleHiddenLayerRegressor max diff:",
      numerical.abs(single_layer_predictions - depth_network_predictions).max())

depth_training_epochs = 200
hidden_layer_width = 32
depth_evaluation_splitter = FoldSplitter(n_splits=5, shuffle=True, random_state=experiment_seed)
depth_random_seeds = [experiment_seed, experiment_seed + 1, experiment_seed + 2]
print(f"\ndepth experiment: width {hidden_layer_width}, adam lr 0.001, 5-fold x 3 seeds, "
      f"{depth_training_epochs} epochs")
print("depth  params  ep(val<=25)  ep(val<=15)  best val  final val  final train  s/run")

depth_error_curves = {}
for hidden_layer_count in [1, 2, 3, 4, 5, 6, 8]:
    validation_error_runs, training_error_runs = [], []
    threshold_25_epochs, threshold_15_epochs = [], []
    evaluation_start_time = performance_clock.time()
    for initialization_seed in depth_random_seeds:
        for training_indices, validation_indices in depth_evaluation_splitter.split(predictor_matrix_float64):
            feature_standardizer = FeatureStandardizer().fit(predictor_matrix_float64[training_indices])
            depth_regressor = VariableDepthRegressor(
                hidden_layer_sizes=(hidden_layer_width,) * hidden_layer_count,
                training_epochs=depth_training_epochs, random_seed=initialization_seed)
            depth_regressor.fit(feature_standardizer.transform(predictor_matrix_float64[training_indices]),
                                house_price_targets_float64[training_indices],
                                feature_standardizer.transform(predictor_matrix_float64[validation_indices]),
                                house_price_targets_float64[validation_indices])
            validation_error_runs.append(depth_regressor.validation_error_history)
            training_error_runs.append(depth_regressor.training_error_history)
            threshold_25_epochs.append(first_epoch_at_or_below(
                depth_regressor.validation_error_history, 25))
            threshold_15_epochs.append(first_epoch_at_or_below(
                depth_regressor.validation_error_history, 15))
    run_count = len(validation_error_runs)
    seconds_per_run = (performance_clock.time() - evaluation_start_time) / run_count
    parameter_count = sum(weight_matrix.size for weight_matrix in depth_regressor.weight_matrices) + sum(
        bias_vector.size for bias_vector in depth_regressor.bias_vectors)
    mean_validation_error_curve = numerical.mean(validation_error_runs, axis=0)
    mean_training_error_curve = numerical.mean(training_error_runs, axis=0)
    depth_error_curves[hidden_layer_count] = (mean_training_error_curve, mean_validation_error_curve)
    threshold_summary_cells = []
    for threshold_epochs in (threshold_25_epochs, threshold_15_epochs):
        threshold_epoch_array = numerical.array(threshold_epochs, dtype=float)
        reached_run_count = int((~numerical.isnan(threshold_epoch_array)).sum())
        median_epoch = (numerical.nanmedian(threshold_epoch_array) if reached_run_count
                        else numerical.nan)
        threshold_summary_cells.append("%3.0f (%2d/%d)" %
                                       (median_epoch, reached_run_count, run_count))
    print("%5d  %6d  %11s  %11s  %8.2f  %9.2f  %11.2f  %5.2f" %
          (hidden_layer_count, parameter_count, threshold_summary_cells[0],
           threshold_summary_cells[1], mean_validation_error_curve.min(),
           mean_validation_error_curve[-1], mean_training_error_curve[-1], seconds_per_run))

figure, (training_axis, validation_axis) = plotting_api.subplots(1, 2, figsize=(11, 4), sharey=True)
for hidden_layer_count, (mean_training_error_curve,
                         mean_validation_error_curve) in depth_error_curves.items():
    training_axis.plot(mean_training_error_curve, label=f"depth {hidden_layer_count}")
    validation_axis.plot(mean_validation_error_curve, label=f"depth {hidden_layer_count}")
training_axis.set_title("Training MSE")
validation_axis.set_title("Validation MSE")
for plot_axis in (training_axis, validation_axis):
    plot_axis.set_yscale("log")
    plot_axis.set_xlabel("epoch")
    plot_axis.axhline(20, color="gray", ls="--", lw=1)
    plot_axis.grid(alpha=0.3)
training_axis.set_ylabel("MSE (mean of 15 runs)")
validation_axis.legend()
plotting_api.tight_layout()
plotting_api.savefig("depth_convergence.png", dpi=150)
plotting_api.close()

print("\nsaved hidden_units.png and depth_convergence.png")
