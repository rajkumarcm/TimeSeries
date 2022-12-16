from lifelines import KaplanMeierFitter
import pandas as pd
from matplotlib import pyplot as plt

#%%
dd = pd.read_csv('dd.csv', header=0)
print(dd.head())
print(dd.describe())

#%%
kmf_dd = KaplanMeierFitter()
continents = dd.un_continent_name.unique()
for continent in continents:
    regimes = dd.loc[dd.un_continent_name==continent, 'regime'].unique()
    plt.figure()
    for regime in regimes:
        event_observed2 = dd.loc[(dd.un_continent_name==continent) &
                                 (dd.regime==regime), 'observed']
        durations2 = dd.loc[(dd.un_continent_name==continent) &
                            (dd.regime==regime), 'duration']
        kmf_dd.fit(durations2, event_observed2, label=regime)
        kmf_dd.plot()
    plt.title(continent)
    plt.tight_layout()
    plt.show()