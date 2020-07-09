import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

sns.set(style = 'dark')

train_data = pd.read_csv("data/train.csv")

sns.relplot(x = 'Age', y = 'Fare', data = train_data, hue = 'Sex')

plt.show()
