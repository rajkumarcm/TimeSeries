import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

main_df = None

ind_cnames = ['Year', 'Day_of_Year', 'Station_Number', 'Air_Temp_Max', 'Air_Temp_Min', 'Air_Temp_Mean',
              'Rel_Humdity_Max', 'Rel_Humidity_Min', 'Rel_Humidity_Mean', 'Vapor_Pressure_Deficit_Mean',
              'Solar_Radiation_Total', 'Precipitation_Total', '4_inch_soil_Max', '4_inch_soil_Min',
              '4_inch_soil_Mean', '20_inch_soil_Max', '20_inch_soil_Min', '20_inch_soil_Mean',
              'wind_speed_Mean',
              'wind_vector_mag', 'wind_vector_dir', 'wind_dir_std', 'max_wind_speed', 'heat_units',
              'reference']
for i in range(2000, 2016+1):
    # if i == 2003:
    #     print('breakpoint...')
    tmp = pd.read_csv(f'Arizona\\{i}.txt', header=None,
                      skip_blank_lines=True)
    tmp = tmp.iloc[:, :len(ind_cnames)].set_axis(ind_cnames, axis=1)

    dates_df = pd.date_range(start=f'{i}-01-01', end=f'{i}-12-31', freq='D')
    dates_df = pd.DataFrame({'Date Local':dates_df, 'Day_of_Year':dates_df.dayofyear})
    tmp = tmp.iloc[:dates_df.shape[0]] # upto max_wind_speed
    tmp['Day_of_Year'] = tmp['Day_of_Year'].astype(int)
    # tmp = tmp.sort_values(by='Day_of_Year', ascending=True, ignore_index=True)
    tmp = pd.merge(left=dates_df, right=tmp, how='left', on='Day_of_Year', sort=False)

    if main_df is None:
        main_df = tmp
    else:
        main_df = pd.concat([main_df, tmp], axis=0)