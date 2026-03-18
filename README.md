# Does Protein Come at a Cost? Analyzing the Taste vs. Health Tradeoff in Recipe Ratings

## Investigation on the Relationship between Protein, Cooking Time, and Ratings of Recipes

**By Sadhana Tadepalli**

### Overview

This is a data science project conducted at UCSD, with a focus on exploring the relationship between the proportion of protein in a recipe, the time it takes to cook it, and the rating over different recipes. 

### Introduction

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

### Data Cleaning and Exploratory Data Analysis

#### Data Cleaning

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
Post merge and newly created columns, the merged dataframe (dubbed `recipes_all`) had 25 columns. I kept the following rows in order to use them later on in the project as they were relevant to my topic: 
After merging, the combined dataframe had 25 columns. I retained only those relevant to the analysis: `name`, `id`, `minutes`, `submitted`, `n_steps`, `n_ingredients`, `rating`, `avg_rating`, `calories`, `protein (PDV)`, `is_high_protein`, and `prop_protein`.

**'recipes_unique' vs 'recipes_all'**
Till now, I had been using the merged dataframe, `recipes_all` which contains multiple rows per recipe since some recipes have multiple ratings. For example, a popular recipe would appear many times. Most analysis using `recipes_all` would be skewed by said popular recipes. Which is why I created `recipes_unique` by dropping the duplicates of recipe ID. `recipes_unique` is the main dataframe I used moving forward, especially considering that most of my analysis was on a recipe-level. In this dataframe, each row has one unqique recipe, has its`avg_rating` and all of the other columns mentioned above.

#### Univariate Analysis
 








 
