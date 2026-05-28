"""Step 1 — Collect a deliberate dataset.

Run:  python 01_collect.py --tag v1

This version cycles through the same fixed seed pool used by 03_benchmark.py
so the model is trained and evaluated on the same environments.

Output:  data_<tag>.npz with arrays `states`, `actions`, `positions`, and
the list `seeds` used during collection.
"""
from __future__ import annotations
import argparse
import threading
import time
import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"
API_KEY = "None"  # paste yours if the server requires it

# Must match the seed pool in 03_benchmark.py
SEED_POOL = [7, 21, 42, 84, 1337]

PHASES = [
    ("Smooth laps",       90, "Hold throttle on straights, smooth steering through corners."),
    ("Tight turns",       60, "Slow before each corner, take it cleanly."),
    ("Obstacle clusters", 60, "Brake when the front ray gets short, steer around."),
    ("Bad terrain",       60, "Drive deliberate lines on ice / mud / sand."),
    ("Recovery",          60, "Drive into walls, get stuck, back out, turn around. DO NOT SKIP."),
]


def _poll_positions(client: GameClient, stop_evt: threading.Event,
                    out: list, hz: float = 5.0):
    """Background thread: poll position at low Hz so we can plot the path later."""
    interval = 1.0 / hz
    while not stop_evt.is_set():
        try:
            st = client.get_latest_state()
            pos = st.get("position") if st else None
            if pos and "x" in pos and "z" in pos:
                out.append((time.time(), pos["x"], pos["z"]))
        except Exception:
            pass
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1",
                    help="Suffix for output file (data_<tag>.npz)")
    args = ap.parse_args()

    client = GameClient(SERVER_URL, API_KEY)

    all_states = []
    all_actions = []
    all_positions = []
    used_seeds = []

    for i, (name, seconds, hint) in enumerate(PHASES, 1):

        # Cycle through the fixed seed pool instead of random seeds
        phase_seed = SEED_POOL[(i - 1) % len(SEED_POOL)]
        used_seeds.append(phase_seed)

        print(f"\n=== Starting phase {i} with seed {phase_seed} ===")

        session = client.create_session(
            mode="time_trial",
            player_name=f"d2w_collector_{args.tag}",
            config={"seed": int(phase_seed), "wind_enabled": False},
        )

        print("Open this URL in a NEW TAB and click into it so WASD reach the game:")
        print(" ", session.get("browser_url"))
        print()

        input("Press Enter once the browser tab has focus and you can see the bot. ")

        client.connect_ws()
        time.sleep(0.5)

        positions = []
        stop_evt = threading.Event()

        t = threading.Thread(
            target=_poll_positions,
            args=(client, stop_evt, positions),
            daemon=True
        )
        t.start()

        client.start_recording(sample_rate=20)

        print(f"\n--- Phase {i}/{len(PHASES)} — {name} ({seconds}s) ---")
        print(f"  {hint}")
        print(f"  Driving for {seconds}s; switch to the browser tab now.")

        for s in range(seconds, 0, -10):
            print(f"  ... {s}s remaining")
            time.sleep(min(10, s))

        stop_evt.set()

        info = client.stop_recording()

        print(f"\nStopped. Samples on the server: {info.get('sample_count', '?')}")

        states_raw, actions = client.get_recording_as_arrays()

        all_states.append(states_raw)
        all_actions.append(actions)

        pos_arr = np.array([(p[1], p[2]) for p in positions], dtype=np.float32)
        all_positions.append(pos_arr)

        try:
            client.disconnect_ws()
            client.delete_session()
        except Exception:
            pass

    states_raw = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    pos_arr = np.concatenate(all_positions, axis=0)

    print(f"\nstates shape   : {states_raw.shape}   (N, 12)")
    print(f"actions shape  : {actions.shape}      (N, 2)")
    print(f"positions shape: {pos_arr.shape}     (M, 2)")

    print(f"\nSeeds used: {used_seeds}")

    assert states_raw.shape[0] >= 5_000, (
        "Fewer than 5,000 samples. Drive more before saving."
    )

    out = f"data_{args.tag}.npz"

    np.savez(
        out,
        states=states_raw,
        actions=actions,
        positions=pos_arr,
        seeds=np.array(used_seeds, dtype=np.int32)
    )

    print(f"Saved {out}")


if __name__ == "__main__":
    main()