```python
import os
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
```


```python
language = 'german'
folder = os.path.join(config.PROCESSED_DATA_DIR, language)
hist = pd.read_csv(os.path.join(folder, 'hist_polysemy_score_german.csv'), sep=';', index_col=0)
contemp = pd.read_csv(os.path.join(folder, 'contemp_polysemy_score_german.csv'), sep=';', index_col=0)
conc = pd.read_csv(os.path.join(folder, 'concreteness_ger.csv'), sep=';', index_col='Word')
```


```python
def get_spearman_corr(df, column, name, conc=conc):
    # create merged dataset
    merged = pd.merge(left=conc, right=df[[column]], left_index=True, right_index=True, how='inner')
    merged.dropna(inplace=True)
    
    # get spearman correlation
    correlation, p_value = spearmanr(merged['concreteness'], merged[column])
    
    # print results
    print(f'Concreteness and {name}')
    print(f'Spearman Correlation Coefficient: {correlation:.4f}')
    print(f'P-value: {p_value:.4e}')
    
    # plot results and save figure
    g = sns.jointplot(x="concreteness", y=column, data=merged, 
                      kind="scatter", joint_kws={"s": 20, "alpha": 0.2})
    fig_name = f"joinplot_concreteness_{column}.png"
    plt.subplots_adjust(left=0.2)
    plt.savefig(os.path.join(config.FIGURES_DIR, fig_name), bbox_inches='tight')
    plt.show()
    
```


```python
get_spearman_corr(hist, 'slope', 'Polysemy Score Evolution')
```

    Concreteness and Polysemy Score Evolution
    Spearman Correlation Coefficient: 0.4173
    P-value: 3.0188e-245
    


    
![png](run_files/run_3_1.png)
    



```python
get_spearman_corr(contemp, 'contemp_polysemy_score', 'Contemporary Polysemy Score')
```

    Concreteness and Contemporary Polysemy Score
    Spearman Correlation Coefficient: 0.0627
    P-value: 2.6267e-12
    


    
![png](run_files/run_4_1.png)
    



```python
get_spearman_corr(hist, 'polysemy_score_1990', 'Historic Polysemy Score 1990s')
```

    Concreteness and Historic Polysemy Score 1990s
    Spearman Correlation Coefficient: -0.0522
    P-value: 6.9536e-08
    


    
![png](run_files/run_5_1.png)
    



```python

```
