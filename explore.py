import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


df = pickle.load(open('collisions.p', 'rb'))

df_yearly = df.groupby('YEAR')
df_monthly = df.groupby('MONTH')
df_hourly = df.groupby('TIME')
df_vehicle = df.groupby('VEHICLE TYPE CODE 1')
df_vehiclecount = df.groupby('vehiclecount')

# some figure parameters
largesize = (16, 9)
smallsize = (7, 7)
titlesize = 30
smalltitle = 20
maincolor = '#009688'
offcolor = '#90A4AE'
showcats = 15  # How many categories to consider
plt.style.use('presentations')  # my custom plot style for presentations



collisions_by_type = df_vehicle.count()['TIME']
injuries_by_car = df_vehicle.mean()['NUMBER OF PERSONS INJURED']
deaths_by_car = df_vehicle.mean()['NUMBER OF PERSONS KILLED']

injuries_by_numcars = df_vehiclecount.mean()['NUMBER OF PERSONS INJURED']
deaths_by_numcars = df_vehiclecount.mean()['NUMBER OF PERSONS KILLED']


# Injury/Death rates by top15 most common vehicles
likelydeaths = pd.DataFrame({'count': df_vehicle.count()['NUMBER OF PERSONS KILLED'],
                             'mean': df_vehicle.mean()['NUMBER OF PERSONS KILLED']})
likelyinju = pd.DataFrame({'count': df_vehicle.count()['NUMBER OF PERSONS INJURED'],
                           'mean': df_vehicle.mean()['NUMBER OF PERSONS INJURED']})
# Remove uninformative categories
uninformative = ['passenger vehicle', 'unknown', 'other']
for vehicle in uninformative:
    injuries_by_car.drop(vehicle, axis=0, inplace=True)
    deaths_by_car.drop(vehicle, axis=0, inplace=True)
    collisions_by_type.drop(vehicle, axis=0, inplace=True)
    likelyinju.drop(vehicle, axis=0, inplace=True)
    likelydeaths.drop(vehicle, axis=0, inplace=True)


fig1, ax1 = plt.subplots(figsize=smallsize)
fig2, ax2 = plt.subplots(figsize=smallsize)
fig4, ax4 = plt.subplots(figsize=smallsize)
fig5, ax5 = plt.subplots(figsize=smallsize)
fig6, ax6 = plt.subplots(figsize=smallsize)
fig7, ax7 = plt.subplots(figsize=smallsize)
fig8, ax8 = plt.subplots(figsize=smallsize)
fig9, ax9 = plt.subplots(figsize=largesize)

# Let's take only the top 15.
top_injuries = injuries_by_car.sort_values().tail(n=showcats)
top_deaths = deaths_by_car.sort_values().tail(n=showcats)
top_deaths_numcar = deaths_by_numcars.sort_values().tail(n=showcats)
top_injuries_numcar = injuries_by_numcars.sort_values().tail(n=showcats)
collisions_by_type = collisions_by_type.sort_values().tail(n=showcats)


likelydeaths = likelydeaths.sort_values(by='count').tail(n=showcats)
likelyinju = likelyinju.sort_values(by='count').tail(n=showcats)

top_deaths.plot.bar(color=maincolor, ax=ax2)
top_injuries.plot.bar(color=maincolor, ax=ax1)
top_injuries_numcar.plot.bar(color=maincolor, ax=ax4)
top_deaths_numcar.plot.bar(color=maincolor, ax=ax5)
collisions_by_type.plot.bar(color=maincolor, ax=ax6)

likelydeaths['mean'].plot.bar(color=maincolor, ax=ax7)
likelyinju['mean'].plot.bar(color=maincolor, ax=ax8)


ax1.set_title('Injuries by vehicle primary type',
              fontsize=smalltitle,
              fontweight='bold')
ax2.set_title('Deaths by vehicle primary type',
              fontsize=smalltitle,
              fontweight='bold')
ax4.set_title('Injuries by # vehicles in collision',
              fontsize=smalltitle,
              fontweight='bold')
ax5.set_title('Deaths by # vehicles in collision',
              fontsize=smalltitle,
              fontweight='bold')
ax6.set_title('Collisions by vehicle type',
              fontsize=smalltitle,
              fontweight='bold')

ax7.set_title('Deaths per collision by likely vehicles',
              fontsize=smalltitle,
              fontweight='bold')

ax8.set_title('Injuries per collision by likely vehicles',
              fontsize=smalltitle,
              fontweight='bold')

for ax in (ax1, ax2, ax4, ax5, ax6, ax7, ax8):
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')


fig1.savefig('injuriesbyvehicle.png', bbox_inches='tight')
fig2.savefig('deathsbyvehicle.png', bbox_inches='tight')
fig5.savefig('deathsbycount.png', bbox_inches='tight')
fig4.savefig('injuriesbycount.png', bbox_inches='tight')
fig6.savefig('collisionsbyvehicle.png', bbox_inches='tight')
fig7.savefig('likelydeaths.png', bbox_inches='tight')
fig8.savefig('likelyinjuries.png', bbox_inches='tight')

fig3, ax3 = plt.subplots(figsize=largesize)
barcolors = [offcolor]*24
barcolors[8] = maincolor
barcolors[9] = maincolor
barcolors[14] = maincolor
barcolors[16] = maincolor
barcolors[17] = maincolor
# Only count accidents where people got injured

data = df_hourly.count()

data['NUMBER OF PERSONS INJURED'].plot.bar(color=barcolors,
                                           ax=ax3)
# ax3.set_xticklabels(rotation=45)

ax3.set_title('Collisions by the hour (2012-present)',
              fontsize=titlesize,
              fontweight='bold',
              loc='left')
fig3.savefig('hourly_counts.png', bbox_inches='tight')

# Dummy encode car types
cars = likelydeaths.index
newdf = df[df['VEHICLE TYPE CODE 1'].isin(cars)]
dummies = pd.get_dummies(newdf['VEHICLE TYPE CODE 1'], prefix_sep='')
newdf = pd.concat([newdf, dummies], axis=1)

# Get correlation
corr = newdf.corr()

# Delete unnecessary labels
newcorr.drop(['UNIQUE KEY', 'YEAR', 'MONTH'], axis=1, inplace=True)
newcorr.drop(['UNIQUE KEY', 'YEAR', 'MONTH'], axis=0, inplace=True)
im = sns.heatmap(newcorr.round(2),
                 cmap='RdBu',
                 annot=False,
                 cbar=True,
                 vmin=-.8,
                 vmax=.8,
                 ax=ax9)
fig9.savefig('heatmap.png', bbox_inches='tight')
