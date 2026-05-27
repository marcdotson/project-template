import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

raw_data = pl.read_csv('../../data/kaggle_car_prices.csv')

# Drop missing values in key columns
df = raw_data.drop_nulls(subset=['make', 'year', 'saledate', 'sellingprice'])

# Clean 'make' column
df = df.with_columns(
    make=pl.col('make').str.to_uppercase()
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