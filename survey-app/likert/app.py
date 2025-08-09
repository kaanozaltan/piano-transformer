import streamlit as st
import pandas as pd
import os
import random

st.set_page_config(page_title="Music Generation Evaluation", layout="centered")

st.title("🎵 Music Generation Evaluation")

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

    # Define criteria
    criteria = {
        "pleasingness": "I enjoy listening to this music. (Pleasingness)",
        "authenticity": "This music sounds natural and human-made. (Authenticity)",
        "novelty": "This music sounds creative and original. (Novelty)",
    }

    if os.path.exists("ratings.csv"):
        ratings_df = pd.read_csv("ratings.csv", dtype={"sample_id": str})
        ratings_df["user_id"] = ratings_df["user_id"].astype(str).str.strip().str.lower()
    else:
        ratings_df = pd.DataFrame(columns=["user_id", "sample_id"] + list(criteria.keys()))

    # --- FILTER ALREADY-RATED SAMPLES ---
    rated_samples = ratings_df.loc[
        ratings_df["user_id"] == user_id, "sample_id"
    ].tolist()

    remaining_samples_df = samples_df[
        ~samples_df["sample_id"].isin(rated_samples)
    ]

    if remaining_samples_df.empty:
        st.success("✅ You have rated all available samples. Thank you!")
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

        likert_options = [
            "Please select a rating",
            "Strongly disagree",
            "Disagree",
            "Neither agree nor disagree",
            "Agree",
            "Strongly agree"
        ]

        likert_map = {
            "Strongly disagree": 1,
            "Disagree": 2,
            "Neither agree nor disagree": 3,
            "Agree": 4,
            "Strongly agree": 5
        }

        for i, sample in enumerate(samples):
            st.subheader(f"🎧 Sample {i+1}")
            st.audio(sample["file_path"])

            sample_id = sample["sample_id"]

            if sample_id not in st.session_state["partial_ratings"]:
                # Initialize empty ratings for this sample
                st.session_state["partial_ratings"][sample_id] = {key: None for key in criteria}

            for key, question in criteria.items():
                current_value = st.session_state["partial_ratings"][sample_id][key]

                # Determine default index for selectbox
                if current_value is None:
                    default_idx = 0
                else:
                    # Find text label corresponding to numeric value
                    label = [label for label, val in likert_map.items() if val == current_value][0]
                    default_idx = likert_options.index(label)

                choice = st.selectbox(
                    f"{question}",
                    likert_options,
                    index=default_idx,
                    key=f"{sample_id}-{key}"
                )

                if choice == "Please select a rating":
                    st.session_state["partial_ratings"][sample_id][key] = None
                else:
                    st.session_state["partial_ratings"][sample_id][key] = likert_map[choice]

        if st.button("Submit Ratings"):
            complete_rows = []
            still_partial = {}

            for sample_id, ratings in st.session_state["partial_ratings"].items():
                if all(value is not None for value in ratings.values()):
                    # Save complete rows
                    row = {
                        "user_id": user_id,
                        "sample_id": sample_id
                    }
                    row.update(ratings)
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
                st.warning("⚠️ No fully completed samples to save.")

            # Keep only incomplete samples in session_state
            st.session_state["partial_ratings"] = still_partial

            st.rerun()
