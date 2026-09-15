"""
league_position_model.py — Technical Team goal-setting model
==================================================================
A deep learning model that predicts a team's future league position
from its last N years of history (league position, Transfermarkt
market value, and market value rank within the league that year).

Architecture: LSTM-based (the input is inherently a time series --
20 years of yearly snapshots per team), not a plain feedforward net.

Input shape:  (sequence_length, 3) per team
              -> [league_position, market_value, market_value_rank_in_league]
              for each of the last `sequence_length` years, oldest year
              first, most recent year last.
Output shape: (1,) -- predicted league position for the following year.

Usage (you handle data collection/vectorization, this just builds/trains):

    import numpy as np
    from league_position_model import build_model, Normalizer

    # X: shape (num_teams, sequence_length, 3), y: shape (num_teams,)
    X_train, y_train = ...   # your 100 teams, years 1-19 -> predicting year 20-ish
    X_test, y_test = ...     # your held-out "last year" split

    normalizer = Normalizer()
    normalizer.adapt(X_train)

    model = build_model(sequence_length=X_train.shape[1], normalizer=normalizer)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=8)

    predictions = model.predict(X_test)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class Normalizer(layers.Normalization):
    """Thin convenience subclass -- just tf.keras.layers.Normalization
    under a clearer name for this file's purpose. Call .adapt(X_train)
    on it before passing it into build_model(), so market value (huge
    numbers) and league position (1-20ish) get put on comparable scales
    internally -- without this, market value dominates the loss purely
    because of its magnitude, regardless of how relevant it actually is.
    """
    pass


def build_model(
    sequence_length: int = 20,
    num_features: int = 3,
    normalizer: "Normalizer | None" = None,
    lstm_units: tuple = (64, 32),
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Builds and compiles the LSTM-based league-position predictor.

    Args:
        sequence_length: how many years of history per team (your "last
            20 years" -> 20).
        num_features: features per year -- 3 for
            [league_position, market_value, market_value_rank].
        normalizer: an already-.adapt()-ed Normalizer layer (recommended
            -- see class docstring above). If None, raw inputs are used
            as-is, which is not recommended given the scale mismatch
            between market value and league position.
        lstm_units: sizes of the two stacked LSTM layers.
        dense_units: size of the dense layer after the LSTM stack.
        dropout_rate: dropout after the dense layer, for regularization
            (100 teams is a small dataset -- overfitting risk is real).
        learning_rate: Adam optimizer learning rate.

    Returns:
        A compiled tf.keras.Model, ready for .fit().
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name="team_history")

    x = inputs
    if normalizer is not None:
        x = normalizer(x)

    # Masking: if some teams have fewer than `sequence_length` years of
    # real history (e.g. promoted/relegated, or founded more recently),
    # pad the missing early years with 0.0 and this layer tells the LSTM
    # to skip those padded timesteps instead of treating them as real
    # (fake) data points.
    x = layers.Masking(mask_value=0.0)(x)

    x = layers.LSTM(lstm_units[0], return_sequences=True)(x)
    x = layers.LSTM(lstm_units[1])(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, name="predicted_league_position")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="league_position_predictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    # Quick manual test with synthetic data, matching your real shape:
    # 100 teams, 20 years, 3 features -- just to confirm the model
    # builds and can actually train a step, not a real result.
    import numpy as np

    np.random.seed(0)
    num_teams, seq_len, num_features = 100, 20, 3

    # Synthetic: league_position (1-20), market_value (1e6-5e8), market_value_rank (1-20)
    positions = np.random.randint(1, 21, size=(num_teams, seq_len, 1)).astype("float32")
    market_values = np.random.uniform(1e6, 5e8, size=(num_teams, seq_len, 1)).astype("float32")
    mv_ranks = np.random.randint(1, 21, size=(num_teams, seq_len, 1)).astype("float32")
    X = np.concatenate([positions, market_values, mv_ranks], axis=-1)
    y = np.random.randint(1, 21, size=(num_teams,)).astype("float32")

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    normalizer = Normalizer()
    normalizer.adapt(X_train)

    model = build_model(sequence_length=seq_len, normalizer=normalizer)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=3, batch_size=8, verbose=2,
    )

    preds = model.predict(X_test, verbose=0)
    print("\nOrnek tahminler (ilk 5):", preds[:5].flatten())
    print("Gercek degerler (ilk 5):", y_test[:5])
