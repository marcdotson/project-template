import numpy as np
import polars as pl
import seaborn.objects as so
import statsmodels.formula.api as smf
import bambi as bmb
import arviz as az

# All U.S. states
states = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

# Approximate probabilities for car sales share by state
# Roughly based on:
# - population
# - vehicle ownership rates
# - economic activity
#
# These do NOT represent real historical data.
# They are simply realistic simulation weights.
probabilities = [
    0.015, # AL
    0.002, # AK
    0.022, # AZ
    0.009, # AR
    0.120, # CA
    0.018, # CO
    0.012, # CT
    0.003, # DE
    0.070, # FL
    0.032, # GA
    0.003, # HI
    0.005, # ID
    0.040, # IL
    0.022, # IN
    0.010, # IA
    0.009, # KS
    0.012, # KY
    0.014, # LA
    0.004, # ME
    0.018, # MD
    0.022, # MA
    0.035, # MI
    0.018, # MN
    0.008, # MS
    0.019, # MO
    0.003, # MT
    0.006, # NE
    0.008, # NV
    0.004, # NH
    0.028, # NJ
    0.007, # NM
    0.065, # NY
    0.033, # NC
    0.003, # ND
    0.038, # OH
    0.013, # OK
    0.014, # OR
    0.040, # PA
    0.003, # RI
    0.016, # SC
    0.003, # SD
    0.021, # TN
    0.085, # TX
    0.009, # UT
    0.002, # VT
    0.026, # VA
    0.020, # WA
    0.005, # WV
    0.018, # WI
    0.002  # WY
]

# Normalize probabilities to ensure they sum to 1
probabilities = np.array(probabilities)
probabilities = probabilities / probabilities.sum()
 

# Set randomization seed
rng = np.random.default_rng(42)

# Specify a function to simulate data
def sim_data(n, beta_0, beta_condition, beta_odometer, beta_vehicle_age, beta_state, sigma):
  # Simulate condition using a normal distribution
  condition = rng.normal(31, 13, size=n)
  # Simulate odometer using a normal distribution
  odometer = rng.normal(68000, 53000, size=n)
  # Simulate vehicle_age (purchase) using a normal distribution
  vehicle_age = rng.normal(60, 20, size=n)
  # Simulate state using a categorical distribution
  state = rng.choice(
    states,
    size=n,
    p=probabilities
  )
  # Normal error term 
  error = rng.normal(0, sigma, size=n)
  
  # Simulate the price outcome
  price = beta_0 + beta_condition * condition + beta_odometer * odometer + beta_vehicle_age * vehicle_age + beta_state * state + error

  # Return the output
  return price, condition, odometer, vehicle_age, state

# Call the function and save as an array
data_arr = sim_data(n = 100, beta_0 = 5000, beta_condition = 32, beta_odometer = 24000, beta_vehicle_age = 20, beta_state = "CA", sigma = 5)

# Convert to a Polars DataFrame
data_df = pl.DataFrame(data_arr, schema = ['price', 'condition', 'odometer', 'vehicle_age', 'state']).to_pandas()

# Fit a frequentist linear regression
fr_fit = smf.ols('price ~ condition + odometer + vehicle_age + state', data = data_df).fit()

# Fit a Bayesian linear regression
ba_fit = bmb.Model('price ~ condition + odometer + vehicle_age + state', data = data_df).fit(progressbar=False)

fr_fit.conf_int() # Interval estimates

az.summary(ba_fit) # Posterior estimates


