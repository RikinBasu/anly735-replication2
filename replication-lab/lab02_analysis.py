"""
ANLY 735 - Replication Laboratory #2
Can the Model Keep Learning?

Proxy replication experiment based on the broader plasticity-loss
phenomenon discussed by Klein et al. (2026).

Research idea:
A model is first trained under Condition A. The input distribution is then
changed by applying a fixed permutation to all input pixels (Condition B).

We compare:

1. WARM MODEL
   - Already trained under Condition A.
   - Continues training under Condition B.

2. FRESH MODEL
   - Newly initialized when Condition B begins.
   - Trained only under Condition B.

The experiment focuses on what happens AFTER the condition changes.

Stability-Plasticity Diagnostic:
CHANGE -> RETENTION -> NEW LEARNING -> RATE -> TRADEOFF

Important:
An immediate performance drop after the condition changes is NOT, by itself,
evidence of plasticity loss. The important comparison is subsequent learning
behavior under Condition B.
"""

# ---------------------------------------------------------------------
# 1. IMPORT PACKAGES
# ---------------------------------------------------------------------

from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------
# 2. EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------
#
# These values are fixed before examining the experimental results.
# Multiple model seeds are used so that our conclusion is not based on
# one lucky or unlucky neural-network initialization.
#

MODEL_SEEDS = [42, 43, 44, 45, 46]

# Controls the train/test split.
SPLIT_SEED = 2026

# Controls the environmental change.
# The SAME pixel permutation is used for every experimental run.
PERMUTATION_SEED = 735

TEST_SIZE = 0.25

# Number of training epochs before and after the environmental change.
PHASE_A_EPOCHS = 40
PHASE_B_EPOCHS = 30

# Number of early post-change epochs used to estimate learning rate.
EARLY_RATE_EPOCHS = 5

# A fixed accuracy threshold used as another measure of adaptation speed.
RECOVERY_THRESHOLD = 0.90

# Neural-network settings.
HIDDEN_UNITS = 64
LEARNING_RATE = 0.05
BATCH_SIZE = 64
ALPHA = 1e-4


# ---------------------------------------------------------------------
# 3. OUTPUT LOCATIONS
# ---------------------------------------------------------------------
#
# This makes the script work even if it is launched from a different
# working directory in VS Code.
#

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"

ANALYSIS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# 4. HELPER FUNCTION: CREATE THE NEURAL NETWORK
# ---------------------------------------------------------------------

def make_model(seed):
    """
    Create a small feed-forward neural network.

    SGD is used with zero momentum to keep the optimization procedure
    relatively simple. The warm and fresh models use the same architecture
    and hyperparameters.

    shuffle=False is intentional because training-example order will be
    controlled manually so the warm and fresh models receive Condition B
    examples in exactly the same order within each experimental run.
    """

    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=LEARNING_RATE,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        momentum=0.0,
        nesterovs_momentum=False,
        shuffle=False,
        random_state=seed,
        max_iter=1,
    )

    return model


# ---------------------------------------------------------------------
# 5. HELPER FUNCTION: TRAIN FOR ONE EPOCH
# ---------------------------------------------------------------------

def train_one_epoch(model, X, y, classes, order):
    """
    Train a model for one pass through the training data.

    partial_fit allows us to continue training the SAME model one epoch
    at a time so that we can measure its learning trajectory.

    The supplied 'order' controls the example order for reproducibility.
    """

    X_epoch = X[order]
    y_epoch = y[order]

    # On the first partial_fit call, scikit-learn must be told all
    # possible target classes.
    if not hasattr(model, "classes_"):
        model.partial_fit(
            X_epoch,
            y_epoch,
            classes=classes,
        )
    else:
        model.partial_fit(
            X_epoch,
            y_epoch,
        )


# ---------------------------------------------------------------------
# 6. HELPER FUNCTION: FIRST EPOCH REACHING A THRESHOLD
# ---------------------------------------------------------------------

def first_epoch_at_threshold(data, accuracy_column, threshold):
    """
    Return the first post-change epoch at which a model reaches the
    requested Condition B accuracy threshold.

    If the threshold is never reached, return NaN.
    """

    matches = data.loc[
        data[accuracy_column] >= threshold,
        "phase_b_epoch",
    ]

    if len(matches) == 0:
        return np.nan

    return int(matches.iloc[0])


# ---------------------------------------------------------------------
# 7. HELPER FUNCTION: EARLY LEARNING RATE
# ---------------------------------------------------------------------

def early_learning_slope(data, accuracy_column, number_of_epochs):
    """
    Estimate the early post-change learning rate.

    A straight line is fitted to accuracy across the first few Condition B
    training epochs. A larger positive slope means accuracy improved more
    rapidly during this early adaptation period.
    """

    subset = data[
        (data["phase_b_epoch"] >= 1)
        & (data["phase_b_epoch"] <= number_of_epochs)
    ]

    slope = np.polyfit(
        subset["phase_b_epoch"],
        subset[accuracy_column],
        1,
    )[0]

    return float(slope)


# ---------------------------------------------------------------------
# 8. LOAD AND PREPARE THE DATA
# ---------------------------------------------------------------------
#
# We use scikit-learn's handwritten Digits dataset.
#
# Each observation is an 8 x 8 grayscale digit image:
#     64 pixel features
#     target classes 0 through 9
#
# Pixel values originally range from 0 to 16.
# Dividing by 16 scales all inputs to approximately [0, 1].
#

digits = load_digits()

X = digits.data.astype(np.float64)
y = digits.target

X = X / 16.0

classes = np.unique(y)

print("Dataset loaded.")
print(f"Total observations: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {classes}")


# ---------------------------------------------------------------------
# 9. CREATE ONE FIXED TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
#
# The same split is used for every run so that variability across runs
# comes primarily from model initialization rather than different data
# partitions.
#

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")


# ---------------------------------------------------------------------
# 10. CREATE CONDITION B: FIXED PIXEL PERMUTATION
# ---------------------------------------------------------------------
#
# CONDITION A:
# Original 64-pixel arrangement.
#
# CONDITION B:
# Exactly the same observations and labels, but all 64 pixel positions
# are rearranged using one fixed permutation.
#
# This creates an input-distribution shift while keeping the prediction
# task (digit classification) and labels unchanged.
#
# Importantly, the same permutation is applied to BOTH training and test
# observations and is identical across all model seeds.
#

permutation_rng = np.random.default_rng(PERMUTATION_SEED)

pixel_permutation = permutation_rng.permutation(X.shape[1])

X_train_B = X_train[:, pixel_permutation]
X_test_B = X_test[:, pixel_permutation]


# Save the exact permutation as part of the reproducibility record.
permutation_df = pd.DataFrame(
    {
        "new_pixel_position": np.arange(len(pixel_permutation)),
        "original_pixel_position_used": pixel_permutation,
    }
)

permutation_df.to_csv(
    ANALYSIS_DIR / "lab02_pixel_permutation.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 11. STORAGE FOR EXPERIMENTAL RESULTS
# ---------------------------------------------------------------------

phase_a_records = []
phase_b_records = []
summary_records = []


# ---------------------------------------------------------------------
# 12. RUN THE EXPERIMENT FOR EACH RANDOM SEED
# ---------------------------------------------------------------------

for seed in MODEL_SEEDS:

    print("\n" + "=" * 60)
    print(f"Running experimental seed: {seed}")
    print("=" * 60)

    # -------------------------------------------------------------
    # PHASE A: TRAIN THE WARM MODEL ON THE ORIGINAL CONDITION
    # -------------------------------------------------------------

    warm_model = make_model(seed)

    for epoch in range(1, PHASE_A_EPOCHS + 1):

        # Create a reproducible training order for this epoch.
        phase_a_rng = np.random.default_rng(
            seed * 10000 + epoch
        )

        training_order = phase_a_rng.permutation(
            len(y_train)
        )

        train_one_epoch(
            warm_model,
            X_train,
            y_train,
            classes,
            training_order,
        )

        # Measure learning on the original Condition A test set.
        condition_a_accuracy = accuracy_score(
            y_test,
            warm_model.predict(X_test),
        )

        phase_a_records.append(
            {
                "seed": seed,
                "phase_a_epoch": epoch,
                "warm_condition_a_accuracy":
                    condition_a_accuracy,
            }
        )

    # -------------------------------------------------------------
    # PERFORMANCE IMMEDIATELY BEFORE THE CHANGE
    # -------------------------------------------------------------

    warm_A_pre_change = accuracy_score(
        y_test,
        warm_model.predict(X_test),
    )

    # -------------------------------------------------------------
    # PERFORMANCE IMMEDIATELY AFTER THE CHANGE
    # -------------------------------------------------------------
    #
    # Notice that NO Condition B training has happened yet.
    #
    # A large drop here only tells us that the environment changed.
    # It does NOT establish plasticity loss.
    # -------------------------------------------------------------

    warm_B_immediate = accuracy_score(
        y_test,
        warm_model.predict(X_test_B),
    )

    print(
        f"Condition A accuracy before change: "
        f"{warm_A_pre_change:.3f}"
    )

    print(
        f"Condition B accuracy immediately after change: "
        f"{warm_B_immediate:.3f}"
    )


    # -------------------------------------------------------------
    # CREATE THE FRESH MODEL
    # -------------------------------------------------------------
    #
    # The fresh model has the same architecture and model seed as the
    # warm model originally had, but has NOT experienced Condition A.
    #
    # This creates a useful paired comparison:
    #
    # warm model  = prior learning history
    # fresh model = no prior learning history
    # -------------------------------------------------------------

    fresh_model = make_model(seed)


    # Store epoch 0 for the warm model.
    # The fresh model has not yet been fitted, so its accuracy is left
    # missing at epoch 0 rather than forcing an arbitrary prediction.
    seed_phase_b_records = [
        {
            "seed": seed,
            "phase_b_epoch": 0,
            "warm_condition_a_accuracy":
                warm_A_pre_change,
            "warm_condition_b_accuracy":
                warm_B_immediate,
            "fresh_condition_a_accuracy":
                np.nan,
            "fresh_condition_b_accuracy":
                np.nan,
        }
    ]


    # -------------------------------------------------------------
    # PHASE B: TRAIN BOTH MODELS ON THE CHANGED CONDITION
    # -------------------------------------------------------------

    for epoch in range(1, PHASE_B_EPOCHS + 1):

        # Both models receive the SAME examples in the SAME order.
        # This helps isolate prior training history as the main
        # difference between them.
        phase_b_rng = np.random.default_rng(
            seed * 20000 + epoch
        )

        training_order = phase_b_rng.permutation(
            len(y_train)
        )

        # Continue training the previously trained model.
        train_one_epoch(
            warm_model,
            X_train_B,
            y_train,
            classes,
            training_order,
        )

        # Train the newly initialized model under the same condition.
        train_one_epoch(
            fresh_model,
            X_train_B,
            y_train,
            classes,
            training_order,
        )


        # ---------------------------------------------------------
        # EVALUATE WARM MODEL
        # ---------------------------------------------------------

        warm_A_accuracy = accuracy_score(
            y_test,
            warm_model.predict(X_test),
        )

        warm_B_accuracy = accuracy_score(
            y_test,
            warm_model.predict(X_test_B),
        )


        # ---------------------------------------------------------
        # EVALUATE FRESH MODEL
        # ---------------------------------------------------------

        fresh_A_accuracy = accuracy_score(
            y_test,
            fresh_model.predict(X_test),
        )

        fresh_B_accuracy = accuracy_score(
            y_test,
            fresh_model.predict(X_test_B),
        )


        seed_phase_b_records.append(
            {
                "seed": seed,
                "phase_b_epoch": epoch,
                "warm_condition_a_accuracy":
                    warm_A_accuracy,
                "warm_condition_b_accuracy":
                    warm_B_accuracy,
                "fresh_condition_a_accuracy":
                    fresh_A_accuracy,
                "fresh_condition_b_accuracy":
                    fresh_B_accuracy,
            }
        )


    # Convert this seed's Phase B results to a DataFrame.
    seed_phase_b_df = pd.DataFrame(
        seed_phase_b_records
    )

    phase_b_records.extend(
        seed_phase_b_df.to_dict("records")
    )


    # -------------------------------------------------------------
    # 13. CREATE SUMMARY METRICS FOR THIS RUN
    # -------------------------------------------------------------

    early_row = seed_phase_b_df.loc[
        seed_phase_b_df["phase_b_epoch"]
        == EARLY_RATE_EPOCHS
    ].iloc[0]

    final_row = seed_phase_b_df.iloc[-1]


    # Mean accuracy across all comparable post-change epochs.
    # Epoch 0 is excluded because the fresh model is not yet fitted.
    comparable_epochs = seed_phase_b_df[
        seed_phase_b_df["phase_b_epoch"] >= 1
    ]

    warm_mean_B_accuracy = comparable_epochs[
        "warm_condition_b_accuracy"
    ].mean()

    fresh_mean_B_accuracy = comparable_epochs[
        "fresh_condition_b_accuracy"
    ].mean()


    summary_records.append(
        {
            "seed": seed,

            # ----- CHANGE -----
            "warm_A_pre_change":
                warm_A_pre_change,

            "warm_B_immediate_after_change":
                warm_B_immediate,

            "immediate_accuracy_change":
                warm_B_immediate - warm_A_pre_change,


            # ----- NEW LEARNING -----
            f"warm_B_epoch_{EARLY_RATE_EPOCHS}":
                early_row["warm_condition_b_accuracy"],

            f"fresh_B_epoch_{EARLY_RATE_EPOCHS}":
                early_row["fresh_condition_b_accuracy"],


            # ----- RATE -----
            "warm_B_early_learning_slope":
                early_learning_slope(
                    seed_phase_b_df,
                    "warm_condition_b_accuracy",
                    EARLY_RATE_EPOCHS,
                ),

            "fresh_B_early_learning_slope":
                early_learning_slope(
                    seed_phase_b_df,
                    "fresh_condition_b_accuracy",
                    EARLY_RATE_EPOCHS,
                ),

            "warm_epochs_to_90pct_B":
                first_epoch_at_threshold(
                    seed_phase_b_df,
                    "warm_condition_b_accuracy",
                    RECOVERY_THRESHOLD,
                ),

            "fresh_epochs_to_90pct_B":
                first_epoch_at_threshold(
                    seed_phase_b_df,
                    "fresh_condition_b_accuracy",
                    RECOVERY_THRESHOLD,
                ),


            # ----- OVERALL CONDITION B LEARNING -----
            "warm_mean_B_accuracy":
                warm_mean_B_accuracy,

            "fresh_mean_B_accuracy":
                fresh_mean_B_accuracy,


            # ----- FINAL CONDITION B PERFORMANCE -----
            "warm_B_final":
                final_row["warm_condition_b_accuracy"],

            "fresh_B_final":
                final_row["fresh_condition_b_accuracy"],

            "warm_minus_fresh_B_final":
                final_row["warm_condition_b_accuracy"]
                - final_row["fresh_condition_b_accuracy"],


            # ----- RETENTION -----
            "warm_A_final_after_B":
                final_row["warm_condition_a_accuracy"],

            "warm_A_retention_change":
                final_row["warm_condition_a_accuracy"]
                - warm_A_pre_change,

            "warm_A_retention_fraction":
                final_row["warm_condition_a_accuracy"]
                / warm_A_pre_change,
        }
    )


# ---------------------------------------------------------------------
# 14. CONVERT RESULTS TO DATAFRAMES
# ---------------------------------------------------------------------

phase_a_df = pd.DataFrame(phase_a_records)

phase_b_df = pd.DataFrame(phase_b_records)

summary_by_seed_df = pd.DataFrame(summary_records)


# ---------------------------------------------------------------------
# 15. CREATE AN OVERALL SUMMARY ACROSS RANDOM SEEDS
# ---------------------------------------------------------------------
#
# Mean and standard deviation allow us to see whether the observed
# learning pattern is reasonably consistent across model initializations.
#

numeric_summary = summary_by_seed_df.drop(
    columns=["seed"]
)

overall_summary_df = (
    numeric_summary
    .agg(["mean", "std"])
    .transpose()
    .reset_index()
)

overall_summary_df.columns = [
    "metric",
    "mean",
    "std",
]


# ---------------------------------------------------------------------
# 16. SAVE NUMERICAL RESULTS
# ---------------------------------------------------------------------

phase_a_df.to_csv(
    ANALYSIS_DIR / "lab02_phase_a_trajectory.csv",
    index=False,
)

phase_b_df.to_csv(
    ANALYSIS_DIR / "lab02_phase_b_trajectory.csv",
    index=False,
)

summary_by_seed_df.to_csv(
    ANALYSIS_DIR / "lab02_summary_by_seed.csv",
    index=False,
)

overall_summary_df.to_csv(
    ANALYSIS_DIR / "lab02_summary_overall.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 17. SAVE THE EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

experiment_config = {
    "dataset": "sklearn.datasets.load_digits",
    "replication_type": "Proxy",
    "model_seeds": MODEL_SEEDS,
    "split_seed": SPLIT_SEED,
    "permutation_seed": PERMUTATION_SEED,
    "test_size": TEST_SIZE,
    "phase_a_epochs": PHASE_A_EPOCHS,
    "phase_b_epochs": PHASE_B_EPOCHS,
    "early_rate_epochs": EARLY_RATE_EPOCHS,
    "recovery_threshold": RECOVERY_THRESHOLD,
    "hidden_units": HIDDEN_UNITS,
    "activation": "relu",
    "optimizer": "SGD",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "alpha": ALPHA,
    "momentum": 0.0,
}

with open(
    ANALYSIS_DIR / "lab02_config.json",
    "w",
    encoding="utf-8",
) as config_file:

    json.dump(
        experiment_config,
        config_file,
        indent=4,
    )


# ---------------------------------------------------------------------
# 18. SAVE COMPUTATIONAL ENVIRONMENT INFORMATION
# ---------------------------------------------------------------------
#
# This directly supports the reproducibility checklist in the QMD.
#

environment_text = f"""
ANLY 735 Replication Laboratory #2
Computational Environment

Python: {sys.version.split()[0]}
Operating System: {platform.platform()}
NumPy: {np.__version__}
pandas: {pd.__version__}
scikit-learn: {sklearn.__version__}
matplotlib: {matplotlib.__version__}
"""

with open(
    ANALYSIS_DIR / "lab02_environment.txt",
    "w",
    encoding="utf-8",
) as environment_file:

    environment_file.write(
        environment_text.strip()
    )


# ---------------------------------------------------------------------
# 19. FIGURE 1 - CONDITION B LEARNING TRAJECTORY
# ---------------------------------------------------------------------
#
# This is the main plasticity figure.
#
# We compare how quickly the WARM and FRESH models learn Condition B.
# Only epochs >= 1 are shown because both models have then received
# Condition B training.
#

condition_b_plot_data = (
    phase_b_df[
        phase_b_df["phase_b_epoch"] >= 1
    ]
    .groupby("phase_b_epoch")
    .agg(
        warm_mean=(
            "warm_condition_b_accuracy",
            "mean",
        ),
        warm_sd=(
            "warm_condition_b_accuracy",
            "std",
        ),
        fresh_mean=(
            "fresh_condition_b_accuracy",
            "mean",
        ),
        fresh_sd=(
            "fresh_condition_b_accuracy",
            "std",
        ),
    )
    .reset_index()
)


plt.figure(figsize=(8, 5))

plt.plot(
    condition_b_plot_data["phase_b_epoch"],
    condition_b_plot_data["warm_mean"],
    label="Warm model",
)

plt.fill_between(
    condition_b_plot_data["phase_b_epoch"],
    condition_b_plot_data["warm_mean"]
    - condition_b_plot_data["warm_sd"],
    condition_b_plot_data["warm_mean"]
    + condition_b_plot_data["warm_sd"],
    alpha=0.20,
)

plt.plot(
    condition_b_plot_data["phase_b_epoch"],
    condition_b_plot_data["fresh_mean"],
    label="Fresh model",
)

plt.fill_between(
    condition_b_plot_data["phase_b_epoch"],
    condition_b_plot_data["fresh_mean"]
    - condition_b_plot_data["fresh_sd"],
    condition_b_plot_data["fresh_mean"]
    + condition_b_plot_data["fresh_sd"],
    alpha=0.20,
)

plt.xlabel("Training epoch after condition change")
plt.ylabel("Condition B test accuracy")

plt.title(
    "Learning Under the Changed Condition"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "lab02_condition_b_learning.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# 20. FIGURE 2 - RETENTION VS ADAPTATION FOR THE WARM MODEL
# ---------------------------------------------------------------------
#
# This figure shows both sides of the stability-plasticity question:
#
# Condition A accuracy = retention / stability
# Condition B accuracy = adaptation / plasticity
#

warm_plot_data = (
    phase_b_df
    .groupby("phase_b_epoch")
    .agg(
        condition_a_mean=(
            "warm_condition_a_accuracy",
            "mean",
        ),
        condition_a_sd=(
            "warm_condition_a_accuracy",
            "std",
        ),
        condition_b_mean=(
            "warm_condition_b_accuracy",
            "mean",
        ),
        condition_b_sd=(
            "warm_condition_b_accuracy",
            "std",
        ),
    )
    .reset_index()
)


plt.figure(figsize=(8, 5))

plt.plot(
    warm_plot_data["phase_b_epoch"],
    warm_plot_data["condition_a_mean"],
    label="Original condition (retention)",
)

plt.fill_between(
    warm_plot_data["phase_b_epoch"],
    warm_plot_data["condition_a_mean"]
    - warm_plot_data["condition_a_sd"],
    warm_plot_data["condition_a_mean"]
    + warm_plot_data["condition_a_sd"],
    alpha=0.20,
)

plt.plot(
    warm_plot_data["phase_b_epoch"],
    warm_plot_data["condition_b_mean"],
    label="Changed condition (new learning)",
)

plt.fill_between(
    warm_plot_data["phase_b_epoch"],
    warm_plot_data["condition_b_mean"]
    - warm_plot_data["condition_b_sd"],
    warm_plot_data["condition_b_mean"]
    + warm_plot_data["condition_b_sd"],
    alpha=0.20,
)

plt.xlabel("Training epoch after condition change")
plt.ylabel("Test accuracy")

plt.title(
    "Warm Model: Retention and New Learning"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "lab02_retention_vs_adaptation.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# 21. DISPLAY FINAL RESULTS IN THE TERMINAL
# ---------------------------------------------------------------------

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print("\nSummary across seeds:")
print(
    overall_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nFiles created:")

print(
    ANALYSIS_DIR
    / "lab02_phase_a_trajectory.csv"
)

print(
    ANALYSIS_DIR
    / "lab02_phase_b_trajectory.csv"
)

print(
    ANALYSIS_DIR
    / "lab02_summary_by_seed.csv"
)

print(
    ANALYSIS_DIR
    / "lab02_summary_overall.csv"
)

print(
    ANALYSIS_DIR
    / "lab02_config.json"
)

print(
    ANALYSIS_DIR
    / "lab02_environment.txt"
)

print(
    FIGURES_DIR
    / "lab02_condition_b_learning.png"
)

print(
    FIGURES_DIR
    / "lab02_retention_vs_adaptation.png"
)

print(
    "\nImportant: interpret plasticity from the POST-CHANGE "
    "learning trajectory, not only from the immediate "
    "performance drop."
)