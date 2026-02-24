import pandas as pd
from collections import defaultdict

# ===== Parameter Settings =====
INPUT_PATH = "filtered_movies.tsv"              # Your input file
OUTPUT_PATH = "final_sample_3000.tsv"           # Output file
MAX_PER_DIRECTOR = 2
QUOTA_PER_GRID = 12                              # Maximum 12 movies per rating-genre-year grid
TARGET_SAMPLE_SIZE = 3000
RATING_REQUIREMENTS = {"Low": 300, "Very High": 300}  # Post-hoc ensure sample size for these rating bins

# ===== Data Loading and Preprocessing =====
df = pd.read_csv(INPUT_PATH, sep="\t")
df = df.dropna(subset=["genres", "averageRating", "startYear", "directors"])
df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
df = df[df["startYear"].between(2021, 2025)]

# Create rating bins
df["rating_bin"] = pd.cut(
    df["averageRating"],
    bins=[0, 5.5, 6.5, 7.5, 10],
    labels=["Low", "Mid", "High", "Very High"]
)

# Split genres into lists
df["genre_list"] = df["genres"].apply(lambda x: x.split(","))

# ===== Initialize Structure =====
selected_movies = []
selected_ids = set()
director_counts = defaultdict(int)

# Get all combination dimensions
rating_bins = df["rating_bin"].dropna().unique()
years = df["startYear"].dropna().unique()
all_genres = sorted({g for genre_list in df["genre_list"] for g in genre_list})

# ===== 3D Cartesian Sampling =====
for rating in rating_bins:
    for genre in all_genres:
        for year in years:
            if len(selected_movies) >= TARGET_SAMPLE_SIZE:
                break

            candidates = df[
                (df["rating_bin"] == rating) &
                (df["startYear"] == year) &
                (df["genre_list"].apply(lambda genres: genre in genres)) &
                (~df["tconst"].isin(selected_ids)) &
                (df["directors"].apply(lambda d: director_counts[d] < MAX_PER_DIRECTOR))
            ]

            sample = candidates.sample(n=min(QUOTA_PER_GRID, len(candidates)), random_state=42)
            for _, row in sample.iterrows():
                if len(selected_movies) >= TARGET_SAMPLE_SIZE:
                    break
                selected_movies.append(row)
                selected_ids.add(row["tconst"])
                director_counts[row["directors"]] += 1

# ===== Convert to DataFrame =====
final_df = pd.DataFrame(selected_movies)

# ===== Post-hoc Correction: Ensure Low and Very High Bin Sample Size =====
rating_counts = final_df["rating_bin"].value_counts().to_dict()

for rating_bin, required_count in RATING_REQUIREMENTS.items():
    current_count = rating_counts.get(rating_bin, 0)
    gap = required_count - current_count

    if gap > 0:
        print(f"⚠️ Post-hoc supplement: Need {gap} more {rating_bin} movies")

        candidates = df[
            (df["rating_bin"] == rating_bin) &
            (~df["tconst"].isin(final_df["tconst"])) &
            (df["directors"].apply(lambda d: director_counts[d] < MAX_PER_DIRECTOR))
        ]

        supplement = candidates.sort_values(
            by="averageRating",
            ascending=(rating_bin == "Low")
        ).head(gap)

        final_df = pd.concat([final_df, supplement], ignore_index=True)

        for d in supplement["directors"]:
            director_counts[d] += 1

# ===== Export Results =====
final_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"✅ Sampling completed, total {len(final_df)} movies, saved to: {OUTPUT_PATH}")

# ===== Sample Rating Structure Overview =====
summary = final_df["rating_bin"].value_counts().sort_index()
print("\n🎯 Rating Distribution:")
print(summary)
