import hashlib
import os
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Musical Turing Test", layout="centered")

st.title("🎵 Musical Turing Test")

# --- USER LOGIN ---
user_id = st.text_input("Please enter your name or participant ID:")

if user_id:
    # Normalize user_id
    user_id = user_id.strip().lower()

    # Check if a new user has logged in → reset shuffled samples
    if st.session_state.get("current_user") != user_id:
        st.session_state["current_user"] = user_id
        st.session_state.pop("shuffled_samples", None)
        st.session_state.pop("partial_ratings", None)

    # --- LOAD DATA ---
    samples_df = pd.read_csv("samples.csv", dtype={"sample_id": str})

    if os.path.exists("ratings.csv"):
        ratings_df = pd.read_csv("ratings.csv", dtype={"sample_id": str})
        ratings_df["user_id"] = ratings_df["user_id"].astype(str).str.strip().str.lower()
    else:
        ratings_df = pd.DataFrame(columns=["user_id", "sample_id", "guess"])

    # --- FILTER ALREADY-RATED SAMPLES ---
    rated_samples = ratings_df.loc[
        ratings_df["user_id"] == user_id, "sample_id"
    ].tolist()

    remaining_samples_df = samples_df[
        ~samples_df["sample_id"].isin(rated_samples)
    ]

    if remaining_samples_df.empty:
        st.success("✅ You have completed all available samples. Thank you!")
    else:
        # --- SHUFFLE ONLY ONCE PER USER ---
        samples = remaining_samples_df.to_dict("records")

        if "shuffled_samples" not in st.session_state:
            st.session_state["shuffled_samples"] = samples.copy()
            random.shuffle(st.session_state["shuffled_samples"])

        samples = st.session_state["shuffled_samples"]

        # --- Initialize partial ratings dict ---
        if "partial_ratings" not in st.session_state:
            st.session_state["partial_ratings"] = {}

        # Instructions
        st.markdown(
            "Please decide for each music excerpt whether you think it was **composed by a human** "
            "or **artificially generated**, or if you are not sure or can't decide, select **not sure**. "
            "Each sample was randomly drawn with equal probability from either human-composed or generated pieces, "
            "distributed equally across the genres Classical, Pop, Soundtrack, Jazz, and Rock."
        )

        # Guess options
        guess_map = {
            "composed by a human": "human",
            "artificially generated": "generated",
            "not sure": "unsure"
        }
        guess_labels = list(guess_map.keys())

        for i, sample in enumerate(samples):
            st.subheader(f"🎧 Sample {i+1}")
            st.audio(sample["file_path"])

            sample_id = sample["sample_id"]

            if sample_id not in st.session_state["partial_ratings"]:
                # Initialize empty ratings for this sample
                st.session_state["partial_ratings"][sample_id] = {"guess": None}

            widget_key = f"{hashlib.md5(sample_id.encode()).hexdigest()}-guess"

            if widget_key not in st.session_state:
                saved = st.session_state["partial_ratings"].get(sample_id, {}).get("guess")
                if saved is not None:
                    label = next((lbl for lbl, val in guess_map.items() if val == saved), None)
                    if label is not None:
                        st.session_state[widget_key] = label

            current_label = st.session_state.get(widget_key, None)
            choice = st.radio(
                "Your choice:",
                guess_labels,
                index=None if current_label is None else guess_labels.index(current_label),
                horizontal=True,
                key=widget_key
            )

            st.session_state["partial_ratings"].setdefault(sample_id, {})["guess"] = guess_map[choice] if choice else None

        if st.button("Submit Answers"):
            complete_rows = []
            still_partial = {}

            for sample_id, ratings in st.session_state["partial_ratings"].items():
                if ratings["guess"] is not None:
                    row = {
                        "user_id": user_id,
                        "sample_id": sample_id,
                        "guess": ratings["guess"]
                    }
                    complete_rows.append(row)
                else:
                    # Keep incomplete samples for next time
                    still_partial[sample_id] = ratings

            if complete_rows:
                new_df = pd.DataFrame(complete_rows)

                if os.path.exists("ratings.csv"):
                    new_df.to_csv("ratings.csv", mode="a", header=False, index=False)
                else:
                    new_df.to_csv("ratings.csv", index=False)

                st.success(f"✅ Saved {len(complete_rows)} completed ratings.")

                # Remove completed samples from shuffled list
                completed_ids = [row["sample_id"] for row in complete_rows]
                new_shuffled = [
                    sample for sample in st.session_state["shuffled_samples"]
                    if sample["sample_id"] not in completed_ids
                ]

                if new_shuffled:
                    st.session_state["shuffled_samples"] = new_shuffled
                else:
                    st.session_state.pop("shuffled_samples", None)

            else:
                st.warning("⚠️ No completed answers to save.")

            # Keep only incomplete samples in session_state
            st.session_state["partial_ratings"] = still_partial

            st.rerun()


if st.query_params.get("secret") == "letmein":
    with open("samples.csv", "rb") as f:
        st.download_button(
            label="Samples",
            data=f,
            file_name="samples.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    with open("ratings.csv", "rb") as f:
        st.download_button(
            label="Ratings",
            data=f,
            file_name="ratings.csv",
            mime="text/csv",
            icon=":material/download:",
        )
