# 30 day LTV prediciton


<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

## <b>Introduction</b>
This project is the result of my team's work in the competition IASA DS Champ, by Institute for Applied System Analysis of Igor Sikorsky Kyiv Polytechnic Institute. Data from a real commercial project were offered for work. The main task was to predict the LifeTime Value for a user based on 7 days of his interaction with the product.

During the work, an exploratory data analysis (<b>`eda.ipynb`</b>) was carried out, which revealed the Poisson nature of most user characteristics, correlations of only certain groups of features with the target, which subsequently helped reduce the dimensionality of the data and keep the model's effectiveness at a satisfactory level. User clustering also greatly improved the results.

In the work (<b>`model.ipynb`</b>), many attempts were made to improve the baseline model, but only the Stacking algorithm brought a significant increase in efficiency.

The final implementation of the algorithm is proposed in the file <b>`model.py`</b>. It trains a model on input data and saves the predicted values to a file.

## <b>Technologies</b>
Project is created with:
- python version: 3.9.12
- numpy version: 1.22.3
- pandas version: 1.4.2
- matplotlib version: 3.5.1
- seaborn version: 0.11.2
- scikit-learn version: 1.1.1

## <b>Launch</b>
In order to use the script, train the model and make a prediction, you need to call the following console command:

```
python model.py train_path test_path answer_path
```

Where:
- train_path - path to file with training data in format .csv
- test_path - path to file with data to predict in format .csv
- answer_path - path to place where predicted data will be stored in format .csv

## <b>Example of use</b>
Out of 1.5 million instances of data, my machine could only learn about a third, producing the following results:

![Example](./result.png)
