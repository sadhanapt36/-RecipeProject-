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
- `prop_protein` — the proportion of a recipe's total calories from protein
- `is_high_protein`, a boolean flag for recipes above the median `prop_protein`
These allow us to compare high and low protein recipes in a standardized way

My analysis is centered around the question of **does prioritizing protein come at a cost to taste?** To answer this question, I looked into whether high-protein recipes are rated diffferently (higher/lower) than low_protein ones, assess whether rating missingness (from the data) depends on the proportion of protein in the recipe, and finally, build a model to predict the cook time of unique recipes, considering that high-protein recipes might involve more cooking time. 
 
