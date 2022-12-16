#%%
from lifelines import KaplanMeierFitter
import pandas as pd
from matplotlib import pyplot as plt

#%% Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header=0)

#%% Q1 - I suppose you meant print the first 5 rows to get an intuition about the data
print(df.head())
print(df.describe())

#%% Q2 - Convert total charges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#%% Q3 - Replace yes and no in the churn column to 1 and 0.
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0 )

#%% Q4 - Replace missing values
df.TotalCharges.fillna(value=df.TotalCharges.median(), inplace=True)

#%% Q5 -
durations = df['tenure']
event_observed = df['Churn']

#%% Q6 - Create kmf object

km = KaplanMeierFitter()

#%% Q7 - Fit the model
km.fit(durations, event_observed, label='Customer Retention')

#%% Q8. - Plot
km.plot()
plt.title('Rate of customer retention over the months')
plt.show()

#%% Q9 - in the report

#%% Q10 - Create Kaplan Meier curves for three cohorts

groups = df.Contract
ix1 = (groups == 'Month-to-month')
ix2 = (groups == 'Two year')
ix3 = (groups == 'One year')

#%% Q11 - Fit Cohort 1, 2, and 3 data and plot the survival curve
kmf = KaplanMeierFitter()
kmf.fit(durations[ix1], event_observed[ix1], label='Month-to-month')
kmf.plot()
kmf.fit(durations[ix2], event_observed[ix2], label='Two year')
kmf.plot()
kmf.fit(durations[ix3], event_observed[ix3], label='One year')
kmf.plot()
plt.title('Rate of customer retention based on different tenure')
plt.tight_layout()
plt.show()

#%% Q12. - in the report

#%% Q14 - include two new cohorts

kmf1 = KaplanMeierFitter()

groups = df.StreamingTV
streaming_no_bindices = (groups == 'No')
streaming_yes_bindices = (groups == 'Yes')

kmf1.fit(durations[streaming_yes_bindices], event_observed[streaming_yes_bindices],
         label='Has StreamingTV')
kmf1.plot()
kmf1.fit(durations[streaming_no_bindices], event_observed[streaming_no_bindices],
         label='Does not have StreamingTV')
kmf1.plot()
plt.title('Rate of customer retention based on StreamingTV subscription')
plt.legend()
plt.tight_layout()
plt.show()

#%% Q17 - Different dataset
dd = pd.read_csv('dd.csv', header=0)
print(dd.head())
print(dd.describe())

#%%
kmf_dd = KaplanMeierFitter()
continents = dd.un_continent_name.unique()
plt.figure()
for continent in continents:
    regimes = dd.loc[dd.un_continent_name==continent, 'regime'].unique()

    event_observed2 = dd.loc[dd.un_continent_name==continent, 'observed']
    durations2 = dd.loc[dd.un_continent_name==continent, 'duration']
    kmf_dd.fit(durations2, event_observed2, label=continent)
    kmf_dd.plot()

plt.title('Survival function of political regime based on different regions')
plt.tight_layout()
plt.show()












