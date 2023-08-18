# This is a test file for something called logistic regression, which is a fancy way of making decisions.
# Let's imagine we're trying to guess if a watermelon is sweet or not based on how it looks.

# ====================== Testing Assumptions for Logistic Regression ======================
# Before we start guessing, we need to check a few things to make sure our guess will be a good one.

# First, we need to gather some tools and materials.
import statsmodels.api as sm
import numpy as np

# Now, we'll use some data about the watermelons. 
# `X` is information about the watermelons, like their size and color. 
# `y` tells us if the watermelon is sweet (1) or not (0).
logit_model = sm.Logit(y, sm.add_constant(X))
result = logit_model.fit()

# We're checking if the size and color of the watermelon can help us predict if it's sweet.
# For each piece of information about the watermelon (like size or color):
for column in X.columns:
    # We calculate something called "log odds" to see how good that piece of information is at guessing.
    log_odds = np.log(result.predict(sm.add_constant(X)) / (1 - result.predict(sm.add_constant(X))))
    # We then see how much that information is related to the "log odds".
    correlation = np.corrcoef(X[column], log_odds)[0, 1]
    # And then we print it out to see the result!
    print(f"Correlation between {column} and log odds: {correlation}")

# Now we're going to check for something called "homoscedasticity". It's a big word, but think of it like this:
# Imagine throwing darts at a target. If you're good, all the darts will be close to the center.
# If the darts are all over the place, then maybe our way of guessing isn't so good.
predicted_vals = result.predict(sm.add_constant(X))
residuals = y - predicted_vals
std_resid = residuals / np.std(residuals)
# This will show us a picture of where our darts landed.
plt.scatter(predicted_vals, std_resid)
plt.axhline(0, color='red')
plt.title("Standardized Residuals vs Predicted values")

# ====================== Checking for Overfitting in Regression ======================
# Now we're checking if we're guessing too specifically based on our current watermelons, 
# so that we might guess wrong for a new watermelon.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# We split our watermelons into two groups: one to practice our guessing and one to test it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# We make our guess based on the practice group.
model = LinearRegression()
model.fit(X_train, y_train)

# We then see how good our guess is for both the practice and test groups.
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

# And print out the results to see!
print(f"Train Score: {train_score}, Test Score: {test_score}")
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")

# ====================== Checking Parts of Time Series Data ======================
# Now we're looking at something called "time series data". Imagine watching a plant grow over time.
# We want to see if the plant grows steadily, if it grows faster in some seasons, and if there are any surprises.
from statsmodels.tsa.seasonal import seasonal_decompose

# `ts` is our plant's growth data over time.
result = seasonal_decompose(ts, model='multiplicative')

# We then look at the steady growth, the seasonal growth, and any surprises.
result.trend.plot(title="Trend Component")
result.seasonal.plot(title="Seasonal Component")
result.resid.plot(title="Residual Component")

# We also want to see if our surprises are just random or if there's a pattern.
from statsmodels.graphics.gofplots import qqplot
qqplot(result.resid.dropna(), line='s')
