import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

make_map = {
  "vw": "volkswagen",
  "mercedes": "mercedes-benz",
  "mercedes-b": "mercedes-benz",
  "landrover": "land rover",
  "ford truck": "ford",
  "ford tk": "ford",
  "chev truck": "chevrolet",
  "gmc truck": "gmc",
  "mazda tk": "mazda",
  "dodge tk": "dodge",
  "hyundai tk": "hyundai",
}

body_map = {
  "regular-cab": "regular cab",

  "supercrew": "crew cab",
  "supercab": "extended cab",
  "crewmax cab": "crew cab",
  "double cab": "crew cab",

  "access cab": "extended cab",
  "club cab": "extended cab",
  "king cab": "extended cab",
  "xtracab": "extended cab",

  "cab plus": "extended cab",
  "cab plus 4": "extended cab",

  "e-series van": "van",
  "promaster cargo van": "van",
  "transit van": "van",
  "ram van": "van",

  "g sedan": "sedan",
  "g coupe": "coupe",
  "g convertible": "convertible",

  "genesis coupe": "coupe",
  "elantra coupe": "coupe",
  "q60 coupe": "coupe",
  "g37 coupe": "coupe",
  "cts coupe": "coupe",
  "cts-v coupe": "coupe",

  "beetle convertible": "convertible",
  "q60 convertible": "convertible",
  "g37 convertible": "convertible",
  "granturismo convertible": "convertible",
  "granturismo convertible": "convertible",

  "cts wagon": "wagon",
  "cts-v wagon": "wagon",
  "tsx sport wagon": "wagon",
}

def value_counts(df, col, **filters):
  for filter_col, filter_val in filters.items():
    df = df.filter(pl.col(filter_col) == filter_val)

  return (
    df.group_by(col)
      .len()
      .sort("len", descending=True)
  )

raw_data = pl.read_csv('../../data/kaggle_car_prices.csv')

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

# Drop missing values in key columns
df = raw_data.drop_nulls(subset=['make', 'body', 'color', 'year', 'saledate', 'sellingprice', 'state', 'transmission'])

# make, body, transmission, state, color need to be consistent and valid

# Clean 'make' column
df = df.with_columns(
  pl.col("make")
  .str.strip_chars()
  .str.to_lowercase()
)

df = df.with_columns(
  pl.col("make")
  .replace(make_map)
)

# Clean up body
df = df.with_columns(
  pl.col("body")
    .str.strip_chars()
    .str.to_lowercase()
)

df = df.with_columns(
  pl.col("body").replace(body_map)
)

# Clean up transmission, convert any not automatic/manual to null and then drop them
df = df.with_columns(
  pl.when(
    ~pl.col("transmission")
      .str.to_lowercase()
      .is_in(["automatic", "manual"])
  )
  .then(None)
  .otherwise(pl.col("transmission").str.to_lowercase())
  .alias("transmission")
)

df = df.drop_nulls(subset=["transmission"])

# CLean up color
df = df.with_columns(
  pl.when(pl.col("color") == "—")
    .then(pl.lit("unknown"))
    .otherwise(pl.col("color"))
    .alias("color")
)

# Extract 'sale_year' from 'saledate'
df = df.with_columns(
  sale_year=pl.col('saledate').str.extract(r'(\d{4})').cast(pl.Int64, strict=False)
)

# Calculate vehicle age: sale_year - year
df = df.with_columns(
  vehicle_age=pl.col('sale_year') - pl.col('year')
)

# Filter out invalid ages
df = df.filter(pl.col('vehicle_age') >= 0)

# Potential states for project 
# │ in    ┆ 3933  │
# │ ne    ┆ 3914  │
# │ sc    ┆ 3882  │
# │ pr    ┆ 2445  │
# │ la    ┆ 2029  │
# │ ut    ┆ 1766  │
# │ ms    ┆ 1730  │
# │ hi    ┆ 1209  │
# │ or    ┆ 1049  │
# │ nm    ┆ 163   │

# Debug unique values, optionally filter by something
# print(value_counts(df, "make", state="ut"))
# print(value_counts(df, "body", state="ut"))
# print(value_counts(df, "transmission", state="ut"))
# print(value_counts(df, "state"))
# print(value_counts(df, "color", state="ut"))

# Setup a 1x3 grid for the 3 plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Automobile Sales Trend Analysis', fontsize=16)

# Plot 1: Most common makes being sold (Top 15)
top_makes = df['make'].value_counts().sort('count', descending=True).head(15).to_pandas()
sns.barplot(data=top_makes, y='make', x='count', hue='make', legend=False, ax=axes[0], palette='magma')
axes[0].set_title('Top 15 Most Common Makes Sold')
axes[0].set_xlabel('Number of Vehicles Sold')
axes[0].set_ylabel('Make')

# Plot 2: Most common years of vehicle sold
common_years = df['year'].value_counts().sort('count', descending=True).head(20).sort('year').to_pandas()
sns.barplot(data=common_years, x='year', y='count', hue='year', legend=False, ax=axes[1], palette='crest')
axes[1].set_title('Most Common Vehicle Years Sold (Top 20)')
axes[1].set_xlabel('Vehicle Year')
axes[1].set_ylabel('Number of Vehicles Sold')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Comparison of vehicle age to selling price
age_price = df.group_by('vehicle_age').agg(
    pl.col('sellingprice').mean().alias('mean_price')
).sort('vehicle_age')
# Filter to a reasonable age range to avoid outliers skewing the plot
age_price = age_price.filter(pl.col('vehicle_age') <= 30).to_pandas()

sns.lineplot(data=age_price, x='vehicle_age', y='mean_price', ax=axes[2], marker='o', color='b')
axes[2].set_title('Average Selling Price by Vehicle Age')
axes[2].set_xlabel('Vehicle Age (Years)')
axes[2].set_ylabel('Average Selling Price ($)')

plt.tight_layout()
plt.savefig('trend_analysis.png')
print("Plots successfully generated and saved to trend_analysis.png")
plt.show()