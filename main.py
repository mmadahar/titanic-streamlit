import polars as pl
import requests
import os.path

# Download the Titanic dataset if it doesn't exist
filename = "titanic.csv"
if not os.path.exists(filename):
    print(f"Downloading Titanic dataset...")
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print("Download complete!")

# Read the CSV using scan_csv
df = pl.scan_csv(filename)

# Group by Survived and Pclass, calculate average age
result = df.group_by(["Survived", "Pclass"]).agg(pl.mean("Age"))

# Collect to materialize the LazyFrame
collected_result = result.collect()

# Pivot the result with Pclass as columns
pivoted = collected_result.pivot(
    index="Survived",
    on="Pclass",
    values="Age"
)

# Print the pivoted results
print(pivoted)
