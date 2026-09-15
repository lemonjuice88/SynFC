"""
league_position_playground.py — YOUR playground, minimal on purpose
=========================================================================
This is deliberately a SCAFFOLD, not a finished model -- the point is
for you to experiment with the architecture yourself. Everything marked
with "# --- DENE / DEĞİŞTİR ---" is meant to be torn out and replaced.

Data contract (matches collect_league_dataset.py's league_dataset.npz):
    history:      shape (num_teams, 21) -- last 21 seasons' final league
                  position, oldest -> newest. Split this yourself into
                  20 input years + 1 held-out target year (or however
                  else you want to slice it -- that's your call).
    market_value: shape (num_teams,) -- current squad value proxy (EUR)

Output: a probability distribution over positions 1-20 (a 20-way
softmax classification, not a single regression number) -- position 1
corresponds to index 0, position 20 to index 19.

This starting scaffold is intentionally simple (no LSTM, just a flat
Dense network) -- swap in whatever you want to try: LSTM, GRU, Conv1D
over the season sequence, attention, deeper/wider Dense stacks, etc.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


NUM_POSITIONS = 20  # league has 20 possible final positions -> 20-way classification


def load_dataset(npz_path: str = "league_dataset.npz"):
    """Loads what collect_league_dataset.py saved. Returns (history, market_value)."""
    data = np.load(npz_path)
    return data["history"], data["market_value"]


def prepare_train_test(history: np.ndarray, market_value: np.ndarray):
    """Splits each team's 21 seasons into 20 input years + 1 held-out
    target year (the most recent one), as you described. Returns
    (X_history, X_market_value, y) ready for the model below.

    y is converted to a 0-indexed class label (position 1 -> class 0,
    ..., position 20 -> class 19) since that's what Keras' sparse
    categorical crossentropy expects.
    """
    X_history = history[:, :20]        # first 20 seasons -> input
    y_position = history[:, 20]        # 21st (most recent) season -> target
    y = (y_position - 1).astype("int32")  # 1..20 -> 0..19

    return X_history, market_value, y


# --- DENE / DEĞİŞTİR: model mimarisi -- burasi tamamen senin oyun alanin ---
def build_playground_model(sequence_length: int = 20) -> tf.keras.Model:
    history_input = layers.Input(shape=(sequence_length,), name="history")
    market_value_input = layers.Input(shape=(1,), name="market_value")

    # --- DENE / DEĞİŞTİR: şu an düz Dense, istersen Reshape edip
    # LSTM/GRU/Conv1D dene (sequence_length, 1) şekline sokman lazım ---
    x = layers.Dense(32, activation="relu")(history_input)
    x = layers.Dense(16, activation="relu")(x)

    # market_value çok büyük sayılar (milyonlarca euro) -- normalize
    # etmeden direkt beslersen agirliklar patlar, log alip kucultuyoruz
    mv = layers.Lambda(lambda v: tf.math.log(v + 1.0))(market_value_input)
    mv = layers.Dense(8, activation="relu")(mv)

    combined = layers.Concatenate()([x, mv])
    combined = layers.Dense(32, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

    outputs = layers.Dense(NUM_POSITIONS, activation="softmax", name="position_probs")(combined)

    model = models.Model(inputs=[history_input, market_value_input], outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
# --- /DENE / DEĞİŞTİR ---


if __name__ == "__main__":
    # Quick manual test with SYNTHETIC data (random), just to confirm
    # the scaffold builds and trains a step without crashing -- swap in
    # your real league_dataset.npz once collect_league_dataset.py has
    # actually run.
    np.random.seed(0)
    num_teams = 100
    fake_history = np.random.randint(1, 21, size=(num_teams, 21)).astype("float32")
    fake_market_value = np.random.uniform(1e6, 5e8, size=(num_teams,)).astype("float32")

    X_hist, X_mv, y = prepare_train_test(fake_history, fake_market_value)

    model = build_playground_model()
    model.summary()

    model.fit(
        {"history": X_hist, "market_value": X_mv}, y,
        epochs=3, batch_size=8, validation_split=0.2, verbose=2,
    )

    probs = model.predict({"history": X_hist[:3], "market_value": X_mv[:3]}, verbose=0)
    for i in range(3):
        predicted_position = probs[i].argmax() + 1
        print(f"Takim {i}: en olasi pozisyon = {predicted_position}, "
              f"olasilik = {probs[i].max():.2%}")
