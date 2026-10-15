from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

random_state = 42
hidden_widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32
gradient_checks = 10
gradient_epsilon = 1e-5
gradient_tolerance = 1e-6
results_dir = Path("results")


digits = load_digits()

# Each image is 8 x 8. sklearn flattens it to 64 features.
# Pixel values are in [0, 16], so divide by 16 to normalize to [0, 1].
X = digits.data.astype(np.float64) / 16.0
y = digits.target


def one_hot_encode(labels, num_classes=10):
    """Convert integer class labels to one-hot encoded rows."""
    encoded = np.zeros((len(labels), num_classes), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


Y = one_hot_encode(y, num_classes=10)

# Split the inputs and both label representations together.
X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X,
    Y,
    y,
    test_size=0.30,
    random_state=random_state,
    stratify=y,
)

print("DATASET")
print("Full dataset:", X.shape)
print("Training inputs:", X_train.shape)
print("Test inputs:", X_test.shape)
print("Training one-hot labels:", Y_train.shape)
print("Test one-hot labels:", Y_test.shape)
print("Normalized input range:", X.min(), "to", X.max())


# ============================================================
# 2. Initialize parameters
# Architecture: 64 -> W -> W -> 10
# Requirement: all weights ~ N(0, 0.01); all biases are zero.
# ============================================================
def initialize_parameters(input_size, hidden_width, output_size=10, seed=42):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0.0, 0.01, size=(input_size, hidden_width)),
        "b1": np.zeros(hidden_width, dtype=np.float64),
        "W2": rng.normal(0.0, 0.01, size=(hidden_width, hidden_width)),
        "b2": np.zeros(hidden_width, dtype=np.float64),
        "W3": rng.normal(0.0, 0.01, size=(hidden_width, output_size)),
        "b3": np.zeros(output_size, dtype=np.float64),
    }


# ============================================================
# 3. Activation functions
# ============================================================
def relu(z):
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0.0, z)


def softmax(logits):
    """Numerically stable softmax, applied independently per row."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
# ============================================================
# 4. Forward propagation
# ============================================================
def forward_pass(X_batch, params):
    # First hidden layer
    Z1 = X_batch @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    # Second hidden layer
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)

    # Output layer: softmax gives one probability per class
    Z3 = A2 @ params["W3"] + params["b3"]
    probabilities = softmax(Z3)

    # Cache values needed by backpropagation
    cache = {
        "X": X_batch,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "P": probabilities,
    }
    return probabilities, cache


# ============================================================
# 5. Cross-entropy loss
# ============================================================
def cross_entropy_loss(probabilities, Y_batch):
    # Y_batch is one-hot, so this selects the probability of each
    # example's correct class. Epsilon prevents log(0).
    epsilon = 1e-12
    correct_class_probabilities = np.sum(Y_batch * probabilities, axis=1)
    return -np.mean(np.log(correct_class_probabilities + epsilon))


# ============================================================
# 6. Backpropagation
# ============================================================
def backward_pass(Y_batch, params, cache):
    m = Y_batch.shape[0]

    X_batch = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    P = cache["P"]

    # Softmax + cross-entropy derivative
    dZ3 = (P - Y_batch) / m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Backpropagate through second ReLU layer
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # Backpropagate through first ReLU layer
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X_batch.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }


# ============================================================
# 7. Accuracy
# ============================================================
def calculate_accuracy(X_data, Y_data, params):
    probabilities, _ = forward_pass(X_data, params)
    predictions = np.argmax(probabilities, axis=1)
    correct_labels = np.argmax(Y_data, axis=1)
    return np.mean(predictions == correct_labels)


# ============================================================
# 8. Numerical gradient checking
# ============================================================
def numerical_gradient(
    params,
    X_batch,
    Y_batch,
    parameter_name,
    index,
    epsilon=gradient_epsilon,
):
    """Central-difference gradient for one parameter."""
    original_value = params[parameter_name][index]

    try:
        _, base_cache = forward_pass(X_batch, params)

        params[parameter_name][index] = original_value + epsilon
        p_plus, cache_plus = forward_pass(X_batch, params)
        loss_plus = cross_entropy_loss(p_plus, Y_batch)

        params[parameter_name][index] = original_value - epsilon
        p_minus, cache_minus = forward_pass(X_batch, params)
        loss_minus = cross_entropy_loss(p_minus, Y_batch)

        # ReLU is not differentiable at zero. A numerical check is valid
        # only when this small perturbation leaves hidden activation masks
        # unchanged. If not, the caller draws another random parameter.
        stable = (
            np.array_equal(base_cache["Z1"] > 0, cache_plus["Z1"] > 0)
            and np.array_equal(base_cache["Z1"] > 0, cache_minus["Z1"] > 0)
            and np.array_equal(base_cache["Z2"] > 0, cache_plus["Z2"] > 0)
            and np.array_equal(base_cache["Z2"] > 0, cache_minus["Z2"] > 0)
        )

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
    finally:
        params[parameter_name][index] = original_value

    return numerical, stable


def run_gradient_check():
    print("\nNUMERICAL GRADIENT CHECK")
    params = initialize_parameters(
        input_size=X_train.shape[1],
        hidden_width=16,
        output_size=Y_train.shape[1],
        seed=random_state,
    )

    # Small fixed batch makes finite-difference checks fast and reproducible.
    X_batch = X_train[:8]
    Y_batch = Y_train[:8]
    probabilities, cache = forward_pass(X_batch, params)
    analytical_gradients = backward_pass(Y_batch, params, cache)

    rng = np.random.default_rng(123)
    parameter_names = list(analytical_gradients.keys())
    passed_results = []
    attempts = 0
    max_attempts = 2000

    while len(passed_results) < gradient_checks and attempts < max_attempts:
        attempts += 1
        name = rng.choice(parameter_names)
        parameter = params[name]
        index = tuple(
            int(rng.integers(0, dimension)) for dimension in parameter.shape
        )

        analytical = analytical_gradients[name][index]
        numerical, stable = numerical_gradient(
            params, X_batch, Y_batch, name, index
        )

        # Do not count points where a ReLU activation changes during the
        # finite-difference perturbation; those points are not differentiable.
        if not stable:
            continue

        denominator = max(
            1e-8, abs(analytical) + abs(numerical)
        )
        relative_error = abs(analytical - numerical) / denominator
        passed = relative_error < gradient_tolerance
        passed_results.append(passed)

        print(
            f"{len(passed_results)}. {name}{index}: "
            f"analytical={analytical:.10e}, "
            f"numerical={numerical:.10e}, "
            f"relative error={relative_error:.3e}, "
            f"PASS={passed}"
        )

    print(f"Stable checks completed: {len(passed_results)}/{gradient_checks}")
    print(f"Passed: {sum(passed_results)}/{len(passed_results)}")
    requirement_met = (
        len(passed_results) == gradient_checks
        and sum(passed_results) >= 9
    )
    print("Gradient-check requirement met:", requirement_met)
    if not requirement_met:
        print(
            "WARNING: the check did not reach 9/10. Review the reported "
            "errors rather than changing the tolerance."
        )


# ============================================================
# 9. Train with mini-batch gradient descent
# ============================================================
def train_network(
    X_train_data,
    Y_train_data,
    hidden_width,
    learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    seed=random_state,
):
    params = initialize_parameters(
        input_size=X_train_data.shape[1],
        hidden_width=hidden_width,
        output_size=Y_train_data.shape[1],
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    loss_history = []

    for epoch in range(epochs):
        # Shuffle the training examples every epoch.
        permutation = rng.permutation(X_train_data.shape[0])
        X_shuffled = X_train_data[permutation]
        Y_shuffled = Y_train_data[permutation]

        for start in range(0, X_train_data.shape[0], batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            probabilities, cache = forward_pass(X_batch, params)
            gradients = backward_pass(Y_batch, params, cache)

            # Gradient-descent update for all weights and biases.
            for name in params:
                params[name] -= learning_rate * gradients[name]

        # Record full-training-set loss after each epoch.
        train_probabilities, _ = forward_pass(X_train_data, params)
        epoch_loss = cross_entropy_loss(train_probabilities, Y_train_data)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Width={hidden_width}, LR={learning_rate:g}, "
                f"epoch={epoch + 1:02d}/{epochs}, loss={epoch_loss:.6f}"
            )

    return params, loss_history


# ============================================================
# 10. Run all combinations: 2 widths x 3 learning rates
# ============================================================
def run_experiments():  
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for width in hidden_widths:
        for learning_rate in learning_rates:
            print(
                f"\nTRAINING EXPERIMENT: width={width}, "
                f"learning_rate={learning_rate:g}"
            )
            start_time = perf_counter()
            params, loss_history = train_network(
                X_train,
                Y_train,
                hidden_width=width,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                seed=random_state,
            )
            training_seconds = perf_counter() - start_time

            train_accuracy = calculate_accuracy(X_train, Y_train, params)
            test_accuracy = calculate_accuracy(X_test, Y_test, params)

            results[(width, learning_rate)] = {
                "params": params,
                "loss_history": loss_history,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_seconds": training_seconds,
            }

            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Test accuracy:     {test_accuracy:.4f}")
            print(f"Training time:     {training_seconds:.2f} seconds")

    # Plot one curve per configuration.
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for (width, learning_rate), result in results.items():
        plt.plot(
            range(1, epochs + 1),
            result["loss_history"],
            label=f"W={width}, lr={learning_rate:g}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Training loss for each width and learning rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = results_dir / "loss_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nLoss curves saved to: {plot_path.resolve()}")

    print("\nEXPERIMENT SUMMARY")
    print(
        f"{'Width':>7} {'Learning rate':>14} {'Train acc.':>12} "
        f"{'Test acc.':>11} {'Time (s)':>10} {'Final loss':>12}"
    )
    for (width, learning_rate), result in results.items():
        print(
            f"{width:>7} {learning_rate:>14.3g} "
            f"{result['train_accuracy']:>12.4f} "
            f"{result['test_accuracy']:>11.4f} "
            f"{result['training_seconds']:>10.2f} "
            f"{result['loss_history'][-1]:>12.6f}"
        )

    return results


# ============================================================
# 11. Benchmark forward and backward passes
# Median of 100 repetitions, as required.
# ============================================================
def benchmark_forward_backward(
    hidden_width=64,
    batch_size=batch_size,
    repetitions=100,
):
    params = initialize_parameters(
        input_size=X_train.shape[1],
        hidden_width=hidden_width,
        output_size=Y_train.shape[1],
        seed=random_state,
    )
    X_batch = X_train[:batch_size]
    Y_batch = Y_train[:batch_size]

    # Warm-up is not included in the timing measurements.
    probabilities, cache = forward_pass(X_batch, params)
    backward_pass(Y_batch, params, cache)

    forward_times = []
    backward_times = []

    for _ in range(repetitions):
        start = perf_counter()
        probabilities, cache = forward_pass(X_batch, params)
        forward_times.append(perf_counter() - start)

        start = perf_counter()
        backward_pass(Y_batch, params, cache)
        backward_times.append(perf_counter() - start)

    print(f"\nTIMING BENCHMARK (median of {repetitions} runs)")
    print(f"Hidden width: {hidden_width}")
    print(f"Batch size: {batch_size}")
    print(f"Forward pass median:  {np.median(forward_times) * 1000:.4f} ms")
    print(f"Backward pass median: {np.median(backward_times) * 1000:.4f} ms")


# ============================================================
# 12. Main program
# ============================================================
if __name__ == "__main__":
    run_gradient_check()
    experiment_results = run_experiments()
    benchmark_forward_backward(
        hidden_width=64,
        batch_size=batch_size,
        repetitions=100,
    )
