import os
# import pdb
import re

import numpy as np
import pandas as pd


# Relative imports
# https://stackoverflow.com/questions/918154/relative-paths-in-python
dirname = os.path.dirname(__file__)
fp = os.path.join(dirname, '../data/gram.csv')

stain = {}
with open(fp, 'r') as file:
    for line in file:
        gram, species = line.strip().split(',')
        stain[species] = gram



class MIC(object):
    '''
    Usage:
    
    # Get all Gram-negative records by only selecting complete MIC profiles
    for _, i in tqdm(df.iterrows()):
        mic = MIC(i, '-')
        if mic.complete:
            break
    '''
    def __init__(self, row, gram=None, expand_mic=False):
        # n = 23, excluded: ceftazidim/ avibactam bc/ only included into
        # profile 2020, so we'd loose a lot of data
        self.gram_negative = sorted(['AMS', 'PIP', 'PIT', 'CXM', 'CTX', 'CAZ', 'AZT', 'IMP', 'MER', 'GEN', 'AMK', 'TOB', 'COL', 'FOS', 'CIP', 'LEV','DOX', 'SXT'])
        # n = 18, excluded: AMP, TIGE, NIT, MECI, MOX, ceftazidim/ avibactam
        self.gram_positive = sorted(['AMS', 'PIT', 'CXM', 'CTX', 'IMP', 'MER', 'GEN', 'AMK', 'FOS', 'CIP', 'LEV', 'MOX', 'ROX', 'DOX', 'SXT', 'CLI', 'VAN', 'TPL', 'RAM', 'LIZ', 'DPT'])
        # n = 21, excluded: AMP, TIGE, PIP
        self.abx = sorted(set(self.gram_positive + self.gram_negative))

        if not gram:
            self.profile, self.log_profile = \
            self.reformat_profile(row[self.abx])

        elif gram == '-':
            self.profile, self.log_profile = \
            self.reformat_profile(row[self.gram_negative])

        elif gram == '+':
            self.profile, self.log_profile = \
            self.reformat_profile(row[self.gram_positive])

        else:
            raise ValueError('This option is not defined')

        self.complete = all([True if not pd.isna(i) else False for i in self.profile])

        if expand_mic:
            # Create a single, Gram + and - MIC and set missing values to 512
            # (meaning "no effect/ resistant/ EUCAST IE/ ...").
            u, v = self.reformat_profile(row[self.abx])
            self.e_profile = np.array(
                [i if not pd.isna(i) else 2 * 256 for i in u])
            self.e_log_profile = np.array(
                [i if not pd.isna(i) else np.log(2 * 256) for i in v])


    def reformat_number(self, n):
        '''
        Cases:
    
        nan, <0.125, >256, "0,25", 0.03125
        '''
        if pd.isna(n):
            return float('nan')
    
        n = n.replace(',', '.')
        
        if '<' in n:
            n = n.replace('<', '')
            return float(n) / 2
            # there are entries such as <0.125 | <2.0 or ...
        elif '>' in n:
            n = n.replace('>', '')
            return 2 * float(n)
        else:
            return float(n)
    
    
    def reformat_profile(self, profile):
        '''
        reformat_profile(dfix.iloc[44][abx], log=True)
        '''
        if all([self.is_valid_number(i) for i in profile]):
            arr = np.array([self.reformat_number(p) for p in profile])
            return (arr, np.log(arr))
        else:
            raise ValueError('There are invalid numbers in the MIC profile')

        
    def is_valid_number(self, n):
        '''
        Example:
    
        valid_numbers = [0.03125, 0.0625, 0.125, 0.25]
        is_valid_number('0.125,0.25', valid_numbers)
        # False
        '''
        if pd.isna(n):
            return True
    
        valid_numbers = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        nr = n.replace(',', '.')
        ns = re.sub('[<>]', '', nr)
        try: 
            return float(ns) in valid_numbers
        except ValueError:
            # Query: '0.125,0.25'
            # ValueError: could not convert string to float: '0.125.0.25'
            return False


    def __repr__(self):
        if self.complete:
            return f'Gram ({gram}) MIC, complete'
        else:
            return f'Gram ({gram}) MIC, incomplete'



class Isolate(object):
    def __init__(self, row, expand_mic=False):
        surname, name, dob, *rest = row
        self.ID = f'{surname}::{name}::{dob}'
        self.species = row['KeimName']
        self.date = pd.to_datetime(row['date'])
        self.ward = row['EinsCode'].split('_')[0]  # F12_EMIS1 > F12
        self.specimen = row['MatCode']
        self.stain = stain.get(self.species)

        mic = MIC(row, self.stain, expand_mic)
        self.mic = mic

    def __repr__(self):
        return (f'{self.species}::{self.date.date().__str__()}::{self.ward}')





