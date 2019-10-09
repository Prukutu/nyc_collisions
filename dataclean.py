import pickle
import difflib
from datetime import datetime

import pandas as pd
import numpy as np


def getYear(timestamp):
    return timestamp.year


def getMonth(timestamp):
    return timestamp.month


def getCarCount(record):
    return record.count()


def combineRedundant(df, catnames):
    df.replace(to_replace=catnames[1:], value=catnames[0], inplace=True)
    return df


def gethour(x):
    return datetime.strptime(x, '%H:%M').hour


df = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv')
weather = pd.read_csv('1411456.csv')

# Convert the DATE column to datetime objects
df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
weather['DATE'] = pd.to_datetime(weather['DATE'], format='%Y-%m-%d')

# Add year and month columns
df['YEAR'] = df['DATE'].apply(getYear)
df['MONTH'] = df['DATE'].apply(getMonth)

# LOCATION column is just the aggregated LAT/LON coordinates, let's remove it.
df.drop(labels='LOCATION', axis='columns')

# Soemthing we can do is combine the 5 "VEHICLE TYPE CODE X" columns into
# a single column containing lists. From there we can remove all nan values
# and then be able to count # of vehicles in each collision.
# df = combineVehicleTypes(df)
cols = ['VEHICLE TYPE CODE ' + str(n) for n in range(1, 6)]
df['vehiclecount'] = df[cols].apply(getCarCount, axis=1)

# Join the snow and precip columns from weather into our dataset
df = df.set_index('DATE').join(weather[['DATE',
                                        'PRCP',
                                        'SNOW']].set_index('DATE'))



# Vehicle types have some redundancies, abbreviations, and mispellings.
# There's also some invalid values like 99999 in the field.
# We find redundancies with difflib (get_close_matches)
suv = ['suv', 'sport utility / station wagon', 'wagon', 'utili',
       'station wagon/sport utility vehicle', 'util', 'ut',]
ambulance = ['ambulance', 'abula', 'am', 'amabu', 'amb', 'ambu', 'ambul',
             'e amb']
fire = ['fire truck', 'fdny', 'fd ny', 'fire truck', 'fire', 'fd tr',
        'firet']
bicycle = ['bicycle', 'bike', 'bicyc']

# Making a judgement call and combining taxis and livery cabs.
# Also, taxis are not a "type" of car rather a car "job"
taxi = ['taxi', 'yello',  'limo', 'limou', 'liver', 'livery vehicle',
        'chassis cab', ]
tanker = ['tanker', 'tank', 'tanke', 'tk', 'tn', 'oil t']
tow = ['tow truck', 'tow', 'tow t', 'tow truck / wrecker', 'tow-t', 'tr']
tractor = ['trac', 'track', 'tract', 'tractor truck diesel',
           'tractor truck gasoline']

# Also grouping all mopeds and motorcycles
motorcycle = ['motorcycle', 'moped', 'mopad', 'mo pa', 'minibike', 'mini',
              'minicycle', 'motor', 'motorbike', 'motorscooter',
              'e sco', 'e-bik', 'scooter']
flatbed = ['flat bed', 'flat', 'flat rack', 'flatb', 'fb', 'bed t']
garbage = ['garba', 'garbage or refuse']
unknown = ['unknown', 'unk', 'unkno']
passenger = ['passenger vehicle', 'pas', 'passa']
pickup = ['pick-up truck', 'pick', 'pick-', 'picku',
          'pickup with mounted camper', 'pk']
armored = ['armored truck', 'ar', 'armor']
bus = ['bus', 'bu']
boxtruck = ['box truck', 'box', 'box t', 'cargo', 'beverage truck']
van = ['van', 'van camper', 'van t', 'van/t', 'vanette', 'vang', 'uhaul', 'vn']

dupeterms = [suv, ambulance, fire, bicycle, taxi, tanker, tow, tractor,
             motorcycle, flatbed, garbage, unknown, passenger, pickup,
             armored, bus, van, boxtruck]



# Then let's replace all the duplicates we found and bad values
for col in cols:
    df[col] = df[col].str.lower()
    df[col].replace('99999', np.nan, inplace=True)

    for term in dupeterms:
        df[col] = combineRedundant(df[col], term)

# To simplify things we'll only keep the hour from the time field.
df['TIME'] = df['TIME'].apply(gethour)

# Dump data into a pickle for later use.
with open('collisions.p', 'wb') as f:
    pickle.dump(df, f)
