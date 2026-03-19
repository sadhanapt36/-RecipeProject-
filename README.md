# Does Protein Come at a Cost? Analyzing the Taste vs. Health Tradeoff in Recipe Ratings

## Investigation on the Relationship between Protein, Cooking Time, and Ratings of Recipes

**By Sadhana Tadepalli**

## Overview

This is a data science project conducted at UCSD, with a focus on exploring the relationship between the proportion of protein in a recipe, the time it takes to cook it, and the rating over different recipes. 

## Introduction

Even though food is a central part of our daily life, for many college students, eating well is a luxury that comes from having extra hours in the day (hours they often don't have). Protein-rich meals such as cooking meats take more time, more steps, and more cooking knowledge than instant meals that are extremeley accessible. As a result, most students default to carbohydrate-heavy meals that are easy and fast to make, sacrificing nutrition for convenience in the process
I was able to acknolwedge that this was something I personally struggle with when my mom and I began talking about improving our diets. Women in my family are more prone to certain health condtions as they age, so we kept circling back to how increasing protein intake should becone a priority. As a self-proclaimed foodie that naturally gravitates toward Indian cuisine and other meals that tend to be carbohydrate-heavy than protein-heavy, **I found myself asking, do protein-packed recipes actually taste good? And do we scarifice flavor by prioritizing protein?**

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

As mentioned, a recipe can have multiple reviews from different users (meaning multiple rows per recipe), I calculated hte mean rating per recipe and merged it back into the merged dataframe. This gives me the `avg_rating` I would be working with throughout the rest of my project.

**4. Split the values in the `nutrition` column into individual float columns**

The `nutrition` column is stored as a string that has list looking content inside (ex. "[138.4, 10.0, 50.0, 3.0, 29.0, 1.0, 6.0]"). I used `ast.literal_eval` to convert each string into an actual list and then split it them into the following columns: 'calories', 'total_fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated_fat (PDV)', 'carbs (PDV).' This was key to the engineering of some of the features I made.

**6. Created `prop_protein` and added it to dataframe**

Rather than using the raw protein PDV value which is not that valuable on its own, I computed the proportion of a recipe's total calories that come from protein, using this formula:
```
prop_protein = (protein_PDV / 100 × 50g × 4 cal/g) / calories
```
The [FDA](https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels#:~:text=Table_title:%20Reference%20Guide:%20Daily%20Values%20for%20Nutrients,Riboflavin%20%7C%20Current%20Daily%20Value:%201.3mg%20%7C) defines 50g as the 100% daily value for protein, and the [USDA](https://www.nal.usda.gov/programs/fnic) says that protein contains 4 calories per gram. This will allow us to compare protein content because by using the PDV value, we wouldn't know the different between a recipe that is 50% or 5% protein.

**7. Added `is_high_protein` to dataframe**

I created a boolean column that checks whether a recipe's `prop_protein` is above it median. I didn't use the mean as the splitting feature due to the heavy skew. This new column was essential in comparing recipes with high and low protein in the hypothesis testing and modeling.

**8. Removed extreme outliers**

Recipes with `minutes > 1440` (recipes that require over 24 hours) and `calories > 5000` were removed as impossible. While there weren't many rows that fit their criteria, if kept in the data, they would have skewed the analysis a lot.

**9. Kept only relevant/necessary columns**

Post dataframe merge and newly created columns, the merged dataframe (dubbed `recipes_all`) had 25 columns. I kept the following rows in order to use them later on in the project as they were relevant to my topic: 
After merging, the combined dataframe had 25 columns. I retained only those relevant to the analysis: `name`, `id`, `minutes`, `submitted`, `n_steps`, `n_ingredients`, `rating`, `avg_rating`, `calories`, `protein (PDV)`, `is_high_protein`, and `prop_protein`.

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

This table is interesting because it really shows the differences between high and low protein that were unclear earlier. High Protein recipes take 20 minutes longer to prepare, 1.5 times more ingredients, and just under 1 more step to make than the low protein recipes. Even here, it is clear that `avg_rating` are almost indeintical by category (4.64 vs. 4.61). This was surprising to me, espeically as someone who came into this project that high-protein recipes require more effort from the cook, but people rate them just as high (so maybe I'm just the lazy one!) as carbohydrate heavy ones. This data showed me, based on this dataset, high-protein recipes don't require giving up good tasting food, but just more of a time and money investment.

What was also helpful was that this aggregate gave me a glimpse into what was coming in the hypothesis testing section.


## Assessment of Missingness

### MNAR Analysis

While there were a couple of columns in the original merged dataset (`recipes_all`) that had a high number of missing values, for the purposes of this section of the project, I chose to focus on the missingness of `reviews` from the `recipes_all` dataframe (has a row for each review), in recipes_unique I dropped this column as it was not directly applicable to the questions I hoped to explore.

Based on what we've seen so far, I believe that the `review` column is **Missing Not at Random (MNAR)**. This closely mirrors my experience with yelp, so I'll use that as an example. When I have a decent, but not excellent or not horrible experience at a restaurant, I'm not likely to go leave a review on Yelp. Similar logic applies here. If someone cooks a recipe and the result is as expected or not that impressive or not dissapointing, then they'll probably not leave a whole review. The missing reviews aren't tried to any other variable, they are more likely to be connected to mediocre, non-standout experiences. This means that missingness is tied to the value that should be recorded (but isn't): mediocrity. That is what makes `reviews` MNAR.

In order to make the missingness of `reviews` **Missing at Random (MAR)**, some additional information that would be helpful is whether or not the user actually cooked the recipe or if they just viewed it. This could be accomplished with a "I made it" button that users can click. By seperating users who viewed the page but didn't cook the recipe and cooked the recipe but didn't leave a review, then we might be able to explain the missingness through a variable like the cooking frequency. 

### Missingness Dependency

Next, I looked at the missingness of `avg_rating` in the dataset. The rows that had `avg_rating` missing in the datafram `recipes_unique` are the recipes that were never reviewed in the first place (which is not considered trivial). I tested whether this missingness depends on `prop_protein` and `sodium (PDV)` and got different results.

 
**Does the missingness of `avg_ratings` (for unique recipes) depend on `prop_protein`?**

Null Hypothesis: The missingness of avg_ratings does not depend on prop_protein (the average proportion of calories from protein is the same for recipes with and without ratings).

Alternate Hypothesis: The missingness of avg_ratings does depend on prop_protein (the average proportion of calories from protein is different for recipes with and without ratings).

Test Statistic: The absolute difference in mean prop_protein between the group with missing avg_rating and the group without missing avg_rating.

Significance Level: 0.05

I ran a permutation test and shuffled the missingness label of `avg_rating` that I named `avg_rating_missing`. I ran the test 1000 times and collected 1000 simulated differences in means. The **observed statistic = -0.0109** is represented by the red line on the graph below.

<iframe src="assets/missingness_perm_prop_protein.html" width="800" height="500" frameborder="0"></iframe>

Through this permutation test, I got a p-value of **0.0** which is < 0.05. Therefore, at my chosen siginficance level, 0.05, I can **reject the null hypothesis**. The missingness of `avg_rating` does depend on `prop_protein`. This result shows that recipes that lack a rating seem to have a slight difference in the protein proportion than those that do have a rating. this suggests that specfic types of recipes, potentially those that are super niche or unfamiliar, are higher in protein and not very likely to get users to post reviews.

**Does the missingness of avg_ratings (for unique recipes) depend on sodium (PDV)?**

Null Hypothesis: The missingness of avg_ratings does not depend on sodium (PDV), the average sodium PDV is the same for recipes with and without ratings

Alternate Hypothesis: The missingness of avg_ratings does depend on sodium (PDV), recipes with missing ratings have a different average sodium PDV content than recipes with ratings

Test Statistic: The absolute difference in mean sodium (PDV) between the group with missing avg_rating and the group without missing avg_rating.

Significance Level: 0.05

Similar to above, I ran a permutation test and shuffled the missingness label of `avg_rating` that I named `avg_rating_missing`. I ran the test 1000 times and collected 1000 simulated differences in means. The **observed statistic = -0.0109** is represented by the red line on the graph below. Interestingly enough, the observed statistic is same for this permutation test and the one above!

<iframe src="assets/missingness_perm_sodium.html" width="800" height="500" frameborder="0"></iframe>

Through this permutation test, I got a p-value of **0.69** which is > 0.05. Therefore, at my chosen siginficance level, 0.05, I  **fail to reject the null hypothesis**. The missingness of `avg_rating` does NOT depend on `prop_protein`. This result intuitively makes sense because a whether a recipe is salty or not has no influence on someone choosing to leave a rating, especially considering that people have very different preferences when it comes to their salting.







 
