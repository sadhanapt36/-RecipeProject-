# Does Protein Come at a Cost? Analyzing the Taste vs. Health Tradeoff in Recipe Ratings

## Investigation on the Relationship between Protein, Cooking Time, and Ratings of Recipes

**By: Sadhana Tadepalli**

## Overview

This is a data science project conducted at UCSD, with a focus on exploring the relationship between the proportion of protein in a recipe, the time it takes to cook it, and the rating over different recipes. 

## Introduction

Even though food is a central part of our daily life, for many college students, eating well is a luxury that comes from having extra hours in the day (hours they often don't have). Protein-rich meals such as cooking meats take more time, more steps, and more cooking knowledge than instant meals that are extremely accessible. As a result, most students default to carbohydrate-heavy meals that are easy and fast to make, sacrificing nutrition for convenience in the process
I was able to acknolwedge that this was something I personally struggle with when my mom and I began talking about improving our diets. Women in my family are more prone to certain health conditions as they age, so we kept circling back to how increasing protein intake should become a priority. As a self-proclaimed foodie that naturally gravitates toward Indian cuisine and other meals that tend to be carbohydrate-heavy than protein-heavy, **I found myself asking, do protein-packed recipes actually taste good? And do we sacrifice flavor by prioritizing protein?**

In order to explore this topic, I analyzed two datasets called, `recipes` and `ratings` pulled from [food.com](https://www.food.com/).

The first dataset, `recipes`, contains **83,782 rows** , which means that there is one row per unique recipe, with the following relevant columns:

| Column | Description |
|---|---|
| `name` | Recipe name |
| `id` | Recipe ID |
| `minutes` | Minutes to prepare recipe |
| `submitted` | Date recipe was submitted |
| `nutrition` | List of [calories, total fat, sugar, sodium, protein, saturated fat, carbs] as % daily value (PDV) |
| `n_steps` | Number of steps in recipe |
| `n_ingredients` | Number of ingredients in recipe |


The second dataset, `interactions`, contains **731,927 rows**, with one per user review and the following relevant columns:

| Column | Description |
|---|---|
| `recipe_id` | Recipe ID |
| `rating` | Rating given (1–5) |

To support my investigation, I extracted `protein (PDV)` from the `nutrition` column and engineered two new features from it: 
- `prop_protein`, the proportion of a recipe's total calories from protein
- `is_high_protein`, a boolean for recipes above the median `prop_protein`
These allow us to compare high and low protein recipes in a standardized way

My analysis is centered around the question of **does prioritizing protein come at a cost to taste?** To answer this question, I looked into whether high-protein recipes are rated diffferently (higher/lower) than low_protein ones, assess whether rating missingness (from the data) depends on the proportion of protein in the recipe, and finally, built a model to predict the cook time of unique recipes, considering that high-protein recipes might involve more cooking time. 

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Before moving forward with any of the analysis I planned for, I did the following to clean the data:

**1. Merged the two datasets**

I merged `recipes` and `interactions` on recipe ID using a left join, so every recipe kept its information even if it had no reviews. The resulting dataframe had one row per user review. Each row contains the recipe informating and the ratings as well. This merged dataframe had multiple rows for some recipes.

**2. Replaced 0 ratings with `NaN`**

The ratings from the `interactions` dataset were on a scale of 1-5. However, there were rows with a rating 0 which means that a user left a comment on the recipe (not relevant for this project) without a star rating. This doesn't actually mean they rated the recipe 0 out of 5, so keeping the value as is would decrease and deflate the ratings, so I replaced the 0s with 'NaN'

**3. Added `avg_rating` per recipe**

As mentioned, a recipe can have multiple reviews from different users (meaning multiple rows per recipe), I calculated the mean rating per recipe and merged it back into the merged dataframe. This gives me the `avg_rating` I would be working with throughout the rest of my project.

**4. Split the values in the `nutrition` column into individual float columns**

The `nutrition` column is stored as a string that has list looking content inside (ex. "[138.4, 10.0, 50.0, 3.0, 29.0, 1.0, 6.0]"). I used `ast.literal_eval` to convert each string into an actual list and then split it them into the following columns: 'calories', 'total_fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated_fat (PDV)', 'carbs (PDV).' This was key to the engineering of some of the features I made.

**5. Created `prop_protein` and added it to dataframe**

Rather than using the raw protein PDV value which is not that valuable on its own, I computed the proportion of a recipe's total calories that come from protein, using this formula:
```
prop_protein = (protein_PDV / 100 × 50g × 4 cal/g) / calories
```
The [FDA](https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels#:~:text=Table_title:%20Reference%20Guide:%20Daily%20Values%20for%20Nutrients,Riboflavin%20%7C%20Current%20Daily%20Value:%201.3mg%20%7C) defines 50g as the 100% daily value for protein, and the [USDA](https://www.nal.usda.gov/programs/fnic) says that protein contains 4 calories per gram. This will allow us to compare protein content because by using the PDV value, we wouldn't know the different between a recipe that is 50% or 5% protein.

**6. Added `is_high_protein` to dataframe**

I created a boolean column that checks whether a recipe's `prop_protein` is above it median. I didn't use the mean as the splitting feature due to the heavy skew. This new column was essential in comparing recipes with high and low protein in the hypothesis testing and modeling.

**7. Removed extreme outliers**

Recipes with `minutes > 1440` (recipes that require over 24 hours) and `calories > 5000` were removed as impossible. While there weren't many rows that fit their criteria, if kept in the data, they would have skewed the analysis a lot.

**8. Kept only relevant/necessary columns**

Post dataframe merge and newly created columns, the merged dataframe (dubbed `recipes_all`) had 25 columns. I kept the following columns in order to use them later on in the project as they were relevant to my topic: 
After merging, the combined dataframe had 25 columns. I retained only those relevant to the analysis: `name`, `id`, `minutes`, `submitted`, `n_steps`, `n_ingredients`, `rating`, `avg_rating`, `calories`, `protein (PDV)`, `saturated_fat (PDV)`, `sodium (PDV)`, `is_high_protein`, and `prop_protein`.

**Distinction between 'recipes_unique' vs 'recipes_all'**

Till now, I had been using the merged dataframe, `recipes_all` which contains multiple rows per recipe since some recipes have multiple ratings. For example, a popular recipe would appear many times. Most analysis using `recipes_all` would be skewed by said popular recipes. Which is why I created `recipes_unique` by dropping the duplicates of recipe ID. `recipes_unique` is the main dataframe I used moving forward, especially considering that most of my analysis was on a recipe-level. In this dataframe, each row has one unqique recipe, has its`avg_rating` and all of the other columns mentioned above.

Here is how the first couple rows of the cleaned version of the merged dataframe, `recipes_unique` looks, it has 82983 rows (82983 unique recipes) and 14 columns:

| name                                 |     id |   minutes | submitted           |   n_steps |   n_ingredients |   rating |   avg_rating |   calories |   protein (PDV) |   saturated_fat (PDV) |   sodium (PDV) | is_high_protein   |   prop_protein |
|:-------------------------------------|-------:|----------:|:--------------------|----------:|----------------:|---------:|-------------:|-----------:|----------------:|----------------------:|---------------:|:------------------|---------------:|
| 1 brownies in the world    best ever | 333281 |        40 | 2008-10-27 00:00:00 |        10 |               9 |        4 |            4 |      138.4 |               3 |                    19 |              3 | False             |           0.04 |
| 1 in canada chocolate chip cookies   | 453467 |        45 | 2011-04-11 00:00:00 |        12 |              11 |        5 |            5 |      595.1 |              13 |                    51 |             22 | False             |           0.04 |
| 412 broccoli casserole               | 306168 |        40 | 2008-05-30 00:00:00 |         6 |               9 |        5 |            5 |      194.8 |              22 |                    36 |             32 | True              |           0.23 |
| millionaire pound cake               | 286009 |       120 | 2008-02-12 00:00:00 |         7 |               7 |        5 |            5 |      878.3 |              20 |                   123 |             13 | False             |           0.05 |
| 2000 meatloaf                        | 475785 |        90 | 2012-03-06 00:00:00 |        17 |              13 |        5 |            5 |      267   |              29 |                    48 |             12 | True              |           0.22 |

### Univariate Analysis
 
**Distribution of `prop_protein`**
 
<iframe src="assets/univariate_prop_protein.html" width="800" height="500" frameborder="0"></iframe>

Based on this graph, its clear that the the distribution of 'prop_protein' is right-skewed seeing as that most of the recipes get less than 20% of their overall calories from protein. This goes to show that high_protein recipes are a minority in this dataset. 

 
**Distribution of `avg_rating`**
 
<iframe src="assets/univariate_avg_rating.html" width="800" height="500" frameborder="0"></iframe>

Based on this graph, it is clear that the distribution of `avg_rating` is very heavily left-skewed with most of the recipes rated somewhere between 4-5 stars. This split in high-rated and low-rated recipes is important in the analysis down the line. Due to the sheer number of recipes in the dataset, the small differences are still important.
 
### Bivariate Analysis
 
**Average Rating compared Protein Level (High vs. Low)**
 
<iframe src="assets/bivariate_protein_avgrating.html" width="800" height="500" frameborder="0"></iframe>

This graph shows how similar the `avg_rating` distrbutions are between the low and high protein recipes. Both groups have a median that is approximately 4.5 stars and similar overall spread, hover over the box plots to see the individual breakdowns. This was the earliest signal for me that protein content may not be a strong indicator or rating considering how similar the two different categories look visually.


**Average Rating vs. Proportion of Calories from Protein**
 
<iframe src="assets/bivariate_density.html" width="800" height="500" frameborder="0"></iframe>

This density heatmap was the most helpful for me in seeing the idea in the box plot, but more closeup. It seems based on this that the highest proportion of recipes are in the 5 stars and low protein section, and the density decreases (the blue lightens) as the proportion of protein increases. There is no clear trend (up or down) in the ratings as `prop_protein` grows which just basically affirmed my idea that there is a weak relationshop between `prop_protein` and `avg_rating`.  

### Interesting Aggregates

For this section, I grouped `recipes_unique` by `is_high_protein`, so there is one row for low protein (False) and high protein (True).

| is_high_protein   |   avg_rating |   calories |   minutes |   n_ingredients |   n_steps |   prop_protein |
|:------------------|-------------:|-----------:|----------:|----------------:|----------:|---------------:|
| False             |      4.64171 |    389.079 |   52.7651 |         8.47264 |   9.71764 |      0.0604602 |
| True              |      4.60842 |    428.516 |   71.7539 |         9.96133 |  10.4102  |      0.255176  |

This table is interesting because it really shows the differences between high and low protein that were unclear earlier. High Protein recipes take 20 minutes longer to prepare, 1.5 more ingredients, and just under 1 more step to make than the low protein recipes. Even here, it is clear that `avg_rating` are almost identical by category (4.64 vs. 4.61). This was surprising to me, espeically as someone who came into this project that high-protein recipes require more effort from the cook, but people rate them just as high (so maybe I'm just the lazy one!) as carbohydrate heavy ones. This data showed me, based on this dataset, high-protein recipes don't require giving up good tasting food, but just more of a time and money investment.

What was also helpful was that this aggregate gave me a glimpse into what was coming in the hypothesis testing section.


## Assessment of Missingness

### MNAR Analysis

While there were a couple of columns in the original merged dataset (`recipes_all`) that had a high number of missing values, for the purposes of this section of the project, I chose to focus on the missingness of `reviews` from the `recipes_all` dataframe (has a row for each review), in recipes_unique I dropped this column as it was not directly applicable to the questions I hoped to explore.

Based on what we've seen so far, I believe that the `review` column is **Missing Not at Random (MNAR)**. This closely mirrors my experience with yelp, so I'll use that as an example. When I have a decent, but not excellent or not horrible experience at a restaurant, I'm not likely to go leave a review on Yelp. Similar logic applies here. If someone cooks a recipe and the result is as expected or not that impressive or not dissapointing, then they'll probably not leave a whole review. The missing reviews aren't tied to any other variable, they are more likely to be connected to mediocre, non-standout experiences. This means that missingness is tied to the value that should be recorded (but isn't): mediocrity. That is what makes `reviews` MNAR.

In order to make the missingness of `reviews` **Missing at Random (MAR)**, some additional information that would be helpful is whether or not the user actually cooked the recipe or if they just viewed it. This could be accomplished with a "I made it" button that users can click. By seperating users who viewed the page but didn't cook the recipe and cooked the recipe but didn't leave a review, then we might be able to explain the missingness through a variable like the cooking frequency. 

### Missingness Dependency

Next, I looked at the missingness of `avg_rating` in the dataset. The rows that had `avg_rating` missing in the datafram `recipes_unique` are the recipes that were never reviewed in the first place (which is not considered trivial). I tested whether this missingness depends on `prop_protein` and `sodium (PDV)` and got different results.

**Does the missingness of `avg_ratings` (for unique recipes) depend on `prop_protein`?**

*Null Hypothesis:* The missingness of avg_ratings does not depend on prop_protein (the average proportion of calories from protein is the same for recipes with and without ratings).

*Alternate Hypothesis:* The missingness of avg_ratings does depend on prop_protein (the average proportion of calories from protein is different for recipes with and without ratings).

*Test Statistic:* The absolute difference in mean prop_protein between the group with missing avg_rating and the group without missing avg_rating.

*Significance Level:* 0.05

I ran a permutation test and shuffled the missingness label of `avg_rating` that I named `avg_rating_missing`. I ran the test 1000 times and collected 1000 simulated differences in means. The **observed statistic = -0.0109** is represented by the red line on the graph below.

<iframe src="assets/missingness_perm_prop_protein.html" width="800" height="500" frameborder="0"></iframe>

Through this permutation test, I got a p-value of **0.0** which is < 0.05. Therefore, at my chosen siginficance level, 0.05, I can **reject the null hypothesis**. The missingness of `avg_rating` does depend on `prop_protein`. This result shows that recipes that lack a rating seem to have a slight difference in the protein proportion than those that do have a rating. this suggests that specfic types of recipes, potentially those that are super niche or unfamiliar, are higher in protein and not very likely to get users to post reviews.


**Does the missingness of avg_ratings (for unique recipes) depend on sodium (PDV)?**

*Null Hypothesis:* The missingness of avg_ratings does not depend on sodium (PDV), the average sodium PDV is the same for recipes with and without ratings

*Alternate Hypothesis:* The missingness of avg_ratings does depend on sodium (PDV), recipes with missing ratings have a different average sodium PDV content than recipes with ratings

*Test Statistic:* The absolute difference in mean sodium (PDV) between the group with missing avg_rating and the group without missing avg_rating.

*Significance Level:* 0.05

Similar to above, I ran a permutation test and shuffled the missingness label of `avg_rating` that I named `avg_rating_missing`. I ran the test 1000 times and collected 1000 simulated differences in means. The **observed statistic = -0.0109** is represented by the red line on the graph below. Interestingly enough, the observed statistic is same for this permutation test and the one above!

<iframe src="assets/missingness_perm_sodium.html" width="800" height="500" frameborder="0"></iframe>

Through this permutation test, I got a p-value of **0.69** which is > 0.05. Therefore, at my chosen siginficance level, 0.05, I  **fail to reject the null hypothesis**. The missingness of `avg_rating` does NOT depend on `sodium (PDV)`. This result intuitively makes sense because a whether a recipe is salty or not has no influence on someone choosing to leave a rating, especially considering that people have very different preferences when it comes to their salting.


## Hypothesis Testing

When I began this project, my initial question was do people rate high-protein recipes differently than low-protein ones? In simpler terms, does adding more protein to a recipe affect the taste (proxy is rating)? To answer this question, I ran a permutation test with the following hypotheses, test statistic, and significance level:

*Null Hypothesis*: Recipes with above-average protein content have the same average rating as recipes with below-average protein content.

*Alternative Hypothesis*: Recipes with above-average protein content have a different average rating than recipes with below-average protein content.

*Test Statistic*: Absolute difference in group means; abs(mean rating of high-protein recipes − mean rating of low-protein recipes)

*Significance Level*: 0.05

I chose a permutation test to answer this question because we aren't assuming any population distribution, we are just checking if the ratings of high_protein recipes and low_protein recipes seem to be from the same distribution. I used `is_high_protein` as the grouping variable. 

I chose the **absolute difference in means** as my test statistic, because I am not looking for a directional difference, my hypothesis is two sided, I am open to either: high-protein recipes are rated higher or lower, or vice versa for low-protein recipes.

To run the permutation test, I shuffled the labels of `is_high_protein` 1000 times to generate test statistics to compare to the observed statistic. The **observed statistic = 0.0333** and is shown on the graph below by the red line.
 
<iframe src="assets/hypothesis_test.html" width="800" height="500" frameborder="0"></iframe>
 
As seen on the graph, the simulated differences are to the far left of the graph while the observed difference is to the far right, which means that a difference this high is pretty unlikely under the null.
 
**p-value:** 0.0
 
### Conclusion

Since the **p-value: 0.0** is < 0.05, the significance level, I **reject the null hypothesis.** The hypothesis test concludes that high and low protein recipes are have different `avg_rating`. However, its important to point out that the observed difference in means is tiny, approximately 0.03 stars. Given the size of this sample, this result was statistically significiant, but practically speaking, a 0.03 difference is next to nothing. 

At the end of the day, hypothesis tests are NOT definitive. We cannot conclude solely from this test alone that protein content (high or low) is what *causes* lower or higher ratings. This hypothesis test largely oversimplifies this situation.


## Framing a Prediction Problem 

Based on the analysis so far, something that stood out to me beyond the main focus of `avg_rating` is that high_protein recipes take almost 20 minutes more to prepare and involve more steps & ingredients (+1 on average) that low-protein recipes. So when choosing a prediction problem, I wanted to see if based on this data, we can predict how long a recipe takes to make based on what we know about it prior to cooking (recipe information we have).

*Prediction Problem:* Predict the cooking time of a recipe.

*Problem Type:* Regression, since `minutes` is a continuous numerical variable.

*Response Variable:* `minutes`, and I chose this because this is one of the most important factors a user may consider in choosing to move forward with a recipe. It also goes back to my initial interest in the topic of the time taken on certain recipes, making them less (or more) accessible to college students trying to eat helathier.

*Evaluation Metric:* BOTH RMSE (root mean squared error) and R²
- RMSE tells us how many minutes off our predictions are on average, so for example, a model with an RMSE of 20 minutes is better than one with a RMSE of 90 minutes.
- R² tells us what proportion of the variance in cook time our model explains. 
- I chose to use both since RMSE and R² give very different pictures of the model. The RMSE on its own doesn't tell us if the model is better than just prediction the mean every single time, and that's where R² comes in. 

**Outlier Removal:**

Before building the model, I looked at the distribution of minutes across ALL recipes, the distribution looks like this:

<iframe src="assets/minutes_before_outliers.html" width="800" height="500" frameborder="0"></iframe>

As you can see, the distribution is extremeley right-skewed, only a tiny group of recipes have cook times of hundred or even thousands of minutes (up till 1400 minutes/24 hours). This probabaly has to do with edge cases such as overnight marination or error in the data entry. Keeping the outliers would make it very difficult to develop a model that can meaningfully learn patterns, since the extreme values would have a large impact on the error calculation and the skew predictions.

In order to address this, I removed outliers using the IQR method. This involved filtering out any recipes where cooking time: `minutes` was lower that Q1-1.5*IQR or higher than Q3+1.5*IQR. The distribution after is show below:

<iframe src="assets/minutes_after_outliers.html" width="800" height="500" frameborder="0"></iframe>

As seen through the graph, the distribution after the outlier removal, is more within reason. This will allow the model to focus on more day-to-day recipes rather than multi-day cooking projects. This change had a dramatic effect on the baseline model's performance, decrease RMSE from approximately 90 minutes to approximately 25 minutes.

**Information Known at Time of Prediction:**
All the features that I used in the model are features that come from recipes_unique or are derived from its columns. The features known at time of prediction include: `prop_protein`, `calories`, `n_steps`, `n_ingredients`, `complexity`, `calories_per_step`. All of these features are available (or can be calculated) before a user has cooked or rated a recipe. I did not use `avg_rating` or `rating` in the model because those are known once the users have interacted with the recipe in some shape or form. Finally, `minutes` is not a features that is used in the prediction, although known since it is the response variable and the one that we are predicting! 


## Baseline Model

For the baseline mode, I used a **Linear Regression** model with two quantitative features: `prop_protein` and `calories`. 

**Features:**

- `prop_protein`, a quantitative variable that I passed through as-is
The proportion of calories from protein might contribute to a higher cooking time because of the type of food it represents. Meats, for example, will take longer to cook than a noodles that cook over the stove. 
- `calories`, a quantitative variable that I passed through as-is
A recipe with higher calories may take more time to cook since, it has more going on. This is a loose conjecture. However, in EDA we saw that `calories` has many high outliers, so I addressed that in the final model.

Based on this model, I calculated the following evaluation metrics:
- **Train R²:** 0.0381
- **Test R²:** 0.0379
- **RMSE:** 24.92 min
 
With these evaluation metrics, I am able to say that the baseline model explains only 3.8% of the variance in cook time. Honestly, this is not a great result, but definitely expected since I only used two features without any transformations being done on them. What stands out to me is the fact that the training and testing R² is almost identical. This is a good sign because it means that the model is not overfitting the training data, its just underfitting all of the data (training and testing). This just tells me that there is a lot of room for growth.
 

## Final Model

For the final model, I settled on a **RandomForestRegressor** and engineered two new features that are able to hone in on the complexity of a recipe, an aspect completely missed in the baseline. I switched to RandomForestRegressor for the final model because it supports hyperparameter tuning and can capture the non-linear relationships between recipe complexity and cook time that Linear Regression cannot. I implemented all of the steps in a single pipeline and used the same `X_train` and `X_test` splits as the baseline for sake of comparision.

**Features from baseline:**

1. `prop_protein` (quantitative), transformed with **QuantileTransformer**. As seen in the univariate analysis, `prop_protein` is a right_skewed distribution, so this transformation turns it into a uniform distribution. I chose to use `prop_protein` instead of `is_high_protein` (which is boolean) because I would have to one-hot encode it, which would reduce the dimensionality of the feature by just splitting protein into two buckets. For example, recipes with 0.58 `prop_protein` and 0.94 `prop_protein` would be in the same bucket of "high protein".

2. `calories` (quantitative), transformed with **RobustScaler**. As mentioned in the baseline section, `calories` is a feature with many outliers, by using RobustScaler, I standardize `calories` by the IQR, making it robust to outliers and extreme values. Even though `calories_per_step` (introduced below) also has `calories` in its calculation, I decided to keep `calories` as its own feature because they are two separate features, fundamentally. `calories` on its own shows the total energy that a recipe can create, while `calories_per_step` breaks down how that energy is distributed across the cooking process. Using both of these features will assist the model in knowing the difference between a high-calorie easy to make dish and a high-calorie hard to make dish.
 
**New engineered features:**
 
1. `calories_per_step` = `calories / n_steps` 
*(quantitative, transformed using StandardScaler)*
 
This new feature was fundamental to the new model because it shows the calories/effort (proxy of steps). Something this feture captures is that lower calories per step might mean a multi-component dish, something `calories` or `n_steps` on their own capture but is essential in predicting cooking time. 
 
2. `complexity` = `n_steps × n_ingredients` 
*(quantitative, transformed using StandardScaler)*
 
This new features show the interaction between the two aspects of a complex recipe: high ingredients and high number of steps (`n_ingredients`, `n_steps`). For example, a dish with 10 steps and 3 ingredients is completely different from one with 10 steps and 10 ingredients. The seocnd dish requires a lot more time and prep, a conclusion you can't come to just by looking at `n_ingredients` or `n_steps` alone. 
 
**Hyperparameter tuning:**

In order to tune the RandomForestRegressor, I chose two hyperparameters worth tuning.

1. `max_depth`: Considering that decision trees are prone to overfitting, limiting the depth allows the model to generalize to data outside of the training data.

2. `n_estimators`: Adding more trees reduces variance by averaging the predictions across more trees improves the predictions accuracy overall.

I tuned these hyperparameters using `GridSearchCV` with a 5-fold cross-validation and used R² as the scoring metric to find the optimal combination and found the optimal parameters: `max_depth = 10` and `n_estimators = 150`.
 
**Evaluation Metrics:**

Based on the final model, I calculated the following evaluation metrics:
- **Baseline Test R²:** 0.0379
- **Final Test R²:** 0.2769
- **Baseline RMSE:** 24.92 min
- **Final RMSE:** 21.61 min

The final model's Test R² improved from 0.038 to 0.277 which means that the R² increased by almost 7 times and the RMSE decreased by approximately 3.5 minutes. The final training R² of 0.352 has some overfitting in comparision to the test R² of 0.2769, but it is understandable due to how complex the Random Forest is. The addition of the complexity based features gave more to the model than just knowing how protein-dense a recipe is. That being said, the model still has room for growth and could definitely benefit from incorporating text analysis of the recipe itself which would give us a clearer picture of how complex it is and potentially explain why high-protein recipes consistently demand more time and effort from the cook (a trend that has been seen consistently).


## Fairness Analysis

For the fairness analysis, keep to the theme of my investigation, I split the recipes into two groups: **high protein (is_high_protein = True)** and **low protein (is_high_protein = False)**. Since I built a regression model, it is best to evaluate using *RMSE parity*, does my model predict cook time (`minutes`) equally well for high and low protein recipes. A model that does worse for one group would be an "unfair" model and would give less reliable cook times to people who are looking for protein-heavy recipes.
 
- **Groups:** high protein (is_high_protein=True) vs low protein (is_high_protein=False) --> used median to split since prop_protein has many outliers

- **Null Hypothesis:** My model is fair, the RMSE for high-protein and low-protein recipes is roughly equal, and any differences are due to random chance.

- **Alternative Hypothesis:** My model is unfair, the RMSE for high_protein recipes is higher than it is for low_protein recipes. 

- **Test Statistic:** Difference in RMSE (high protein - low protein)

- **Significant Level:** 0.05
 

I ran a permutation test and shuffled `is_high_protein` labels 1000 times, and compared the differences to the observed statistic. The observed statistic is shown by the red vertical line below. 
 

<iframe src="assets/fairness_analysis.html" width="800" height="500" frameborder="0"></iframe>
 

The observed statistic is indicated by the red vertical line. The RMSE for high-protein recipes was 21.9412 minutes and for low-protein recipes was 21.3534 minutes, giving an observed difference of approximately 0.58 minutes.
 
**p-value:** 0.022

Since the p-value is **less than 0.05**, we **reject the null hypothesis**. The permutation test suggests that the final model is unfair, that it predicts cook time less accurately for high-protein recipes than for low-protein recipes. That being said, it is important to point out that a difference (in the observed statistic) of ~ 0.58 is basically negligible. This has come up throughout, but due to the sheer size of the datast, the tests are able to detect minute differences as statistically significiant. While I am not dismissing the results of this fairness analysis, I do think we should note that the impact of this difference is next to nothing since the model performs almost exactly the same on both groups. 

 

 
 