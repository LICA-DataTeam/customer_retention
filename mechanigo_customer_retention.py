# import required modules
import pandas as pd
import numpy as np
import os, sys
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

import matplotlib.pyplot as plt
import re, string
import seaborn as sns
#import datetime as dt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes import GammaGammaFitter
from joblib import dump, load

pd.options.mode.chained_assignment = None  # default='warn'

## INITIALIZATION
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path) # current working directory

def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()

def fix_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  
  Parameters:
  -----------
  name: str
    
  Returns:
  --------
  fixed name; str
  
  '''
  name_corrections = {'MERCEDESBENZ' : 'MERCEDES-BENZ',
                      'MERCEDES BENZ' : 'MERCEDES-BENZ',
                      'IXSFORALL INC.' : 'Marvin Mendoza (iXSforAll Inc.)',
                      'GEMPESAWMENDOZA' : 'GEMPESAW-MENDOZA',
                      'MIKE ROLAND HELBES' : 'MIKE HELBES'}
  name_list = list()
  # removes emojis and ascii characters (i.e. chinese chars)
  name = remove_emoji(name).encode('ascii', 'ignore').decode()
  # split name by spaces
  for n in name.split(' '):
    if n not in name_list:
    # check each character for punctuations minus '.' and ','
        name_list.append(''.join([ch for ch in n 
                                if ch not in string.punctuation.replace('.', '')]))
    else:
        continue
  name_ = ' '.join(name_list).strip().upper()
  for corr in name_corrections.keys():
      if re.search(corr, name_):
          return name_corrections[re.search(corr, name_)[0]]
      else:
          continue
  return name_

def fix_phone_num(phone):
    '''
    Cleanup phone numbers
    
    Parameters
    ----------
    phone : string
    
    Returns:
    phone : string
    '''
    phone = ''.join(phone.split(' '))
    phone = ''.join(phone.split('-'))
    phone = re.sub('^\+63', '0', phone)
    phone = '0' + phone if phone[0] == '9' else phone
    return phone

def fix_address(address):
    cities = {'CALOOCAN' : 'CALOOCAN', 'LAS PIÑAS' : 'LAS PIÑAS', 
              'LAS PINAS' : 'LAS PINAS','MAKATI' : 'MAKATI', 
              'MALABON' : 'MALABON', 'MANDALUYONG' : 'MANDALUYONG', 
              'MANILA' : 'MANILA', 'MARIKINA' : 'MARIKINA', 
              'MUNTINLUPA' : 'MUNTINLUPA', 'NAVOTAS' : 'NAVOTAS', 
              'PARAÑAQUE' : 'PARAÑAQUE', 'PARANAQUE' : 'PARAÑAQUE',
              'PASAY' : 'PASAY', 'PASIG' : 'PASIG', 
              'PATEROS' : 'PATEROS', 'Q(UEZON)?.*C(ITY)?' : 'QUEZON',
              'SAN JUAN' : 'SAN JUAN', 'TAGUIG' : 'TAGUIG', 
              'VALENZUELA' : 'VALENZUELA'}
    provinces = {'RIZAL' : 'RIZAL', 'LAGUNA' : 'LAGUNA', 'BULACAN' : 'BULACAN',
                'CAVITE' : 'CAVITE' , 'TAYTAY' : 'RIZAL', 'TARLAC' : 'TARLAC',
                'NUEVA ECIJA' : 'NUEVA ECIJA', 'PAMPANGA' : 'PAMPANGA',
                'BATANGAS' : 'BATANGAS', 'QUEZON' : 'QUEZON', 'BATAAN' : 'BATAAN',
                'AURORA' : 'AURORA', 'ZAMBALES' : 'ZAMBALES'}
    
    if pd.isna(address):
        return ''
    else:
        address_clean = ''
        for city in cities.keys():
            if re.search(city, address.upper()):
                address_clean = f'{cities[city]} CITY, METRO MANILA, PHILIPPINES'
            else:
                continue
        # check if city dict has match
        if len(address_clean) == 0:
            for prov in provinces.keys():
                if re.search(prov, address.upper()):
                    address_clean = f'{provinces[prov]}, PHILIPPINES'
                else:
                    continue
        else:
            return address_clean
        # no match found
        if len(address_clean) == 0:
            address_clean = address
    
        return address_clean
        
            
def get_ratio(a, b):
  return a/b if b else 999


def model_year_group(year):
    '''
    Returns grouping for car model year
    '''
    if year <= 1996:
        return '-1996'
    elif year >= 1997 and year <= 2005:
        return '1997-2005'
    elif year >= 2006 and year <= 2015:
        return '2006-2015'
    elif year >= 2016 and year <= 2020:
        return '2016-2020'
    else:
        return '2021-'

service_category_dict = {'PMS': ['(PMS|(BIRTHDAY PROMO)|(13TH MONTH PROMO)|(BONUS OFFER))',
                            'ANNUAL.*PMS PACKAGE', 'BASIC OIL CHANGE (.*)'],
                    'DISINFECTION': ['CAR.*((SANITATION (AND|&)?)|(FOGGING))', 'DISINFECT'],
                    'AIRCON': ['(AC|AIRCON) CLEANING', '(AC|AIRCON) BELT', '(AIR (AND )?|CABIN){1,2} FILTER',
                               'FREON CHARGING'],
                    'TIRES': ['VULCANIZING', '(TIRE(S)?|WHEEL) ALIGNMENT', 'TIRE INSTALLATION',
                              'TIRE BALANCING', 'TIRE ROTATION', 'TIRE MOUNTING',
                              'TIRE INFLAT(E|ION)', 'SPARE TIRE', 'TIRE(S)?',
                              'YOKOHAMA', 'GOODYEAR', 'BF(\s)?GOODRICH', 'WESTLAKE',
                              'AC(\s)DELCO(\s)?NS[0-9]0'],
                    'LIGHTS': ['LIGHT', 'BULB'],
                    'BATTERIES': ['BATTERY', 'AC(\s)?DELCO(\s)?[A-Z]{3}([- ]?[A-Z]{1})?[0-9]{2}'],
                    'BRAKES': ['BRAKE', 'CALIPER'],
                    'DIAGNOSIS & INSPECTION': ['ENGINE SCANNING', 'CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION'],
                    'MAINTENANCE': ['(OIL )?(CHANGE|REPLACE|PREMIUM PACKAGE)( OIL)?[-\s]?(GAS|DIESEL|MINERAL OIL|GEAR OIL)?(\s)?((SEMI|FULLY)(-)? SYNTHETIC|REGULAR)?(\s)?(.*)?', 
                                    'COOLANT(\s)?([0-9]L|DRAIN AND REFILL)?', 'ATF (DIALYSIS|DRAIN AND REFILL|FLUID)', 'SPARK PLUG',
                                    'DRIVE BELT', 'ENGINE FLUSHING', 'WIPER BLADE(\s)?(ASSEMBLY)?', '(IRIDIUM)?(\s)?SPARK(\s)?PLUG(S)?( OIL SEAL)?', 
                                    'COIL PACK', 'OIl FILTER', 'VORTEX PLUS ENGINE OIL ([0-9]+ LITER(S)?)',
                                    'VALVE COVER GASKET', 'CRANK(\s)?SHAFT (OIL SEAL|PULLEY)', 'TRANSMISSION PAN GASKET', 
                                    'OIL GASKET SPOOL VALVE', 'AUTOMATIC TRANSMISSION FLUSH', 'SENDING OIL SENSOR',
                                    'SHELL OIL [0-9]L', 'TOYOTA\s?(OIL)?\s?5W[0-9]0\s(GENUINE OIL)?',
                                    'HONDA (FULLY|SEMI) SYNTHETIC OIL [0-9]W[0-9]0', 'SHOCK ABSORBERS',
                                    'ALTERNATOR', 'ENGINE HOSE', 'FUEL (HOSE\s)?TANK(\sHOSE)?', 'FAN MOTOR', 
                                    'POWER STEERING PUMP KIT', 'THERMOSTAT HOUSING', 'ENGINE (FLUSHING|SUPPORT)', 
                                    'WINDSHIELD WIPER SET', 'STABILIZER LINK', 'RADIATOR FAN MOTOR', 'ABS SENSOR',
                                    'FUEL FILTER', '(TUNE|CHECK)[- ]?UP', 'FUEL PUMP', 'MAGNETIC CLUTCH', 'TINT', 'CLUTCH',
                                    'TENSIONER BEARING', 'BALL JOINT', 'DOOR HANDLE', 'STARTER ASSEMBLY', 'CVT FLUID',
                                    'MASS AIR FLOW SENSOR']
                    }

def get_service_category(x):
    for cat in service_category_dict.keys():
        for service in service_category_dict[cat]:
            if re.search(service, str(x).upper()):
                return cat
            else:
                continue
    return 'MISCELLANEOUS'


tire_brands = ["BRIDGESTONE", "DUNLOP", "GOODYEAR", "MICHELIN", "BF(\s)?GOODRICH",
               "MAXXIS", "NEXEN", "NITTO", "TOYO", "YOKOHAMA", "ALLIANCE", "ARIVO", 
               "DOUBLE(\s)?COIN", "TORQUE", "WANLI", "COOPER", "CST", "PRESA", "WESTLAKE"]

def get_service_name(service_category, service_name):
    cat = service_category
    service_name = service_name.upper()
    service = []
    # LIGHTS
    if cat == 'LIGHTS':
        lights = ['HEAD', 'TAIL', 'PARK', 'BRAKE', 'PLATE', 'SIGNAL', 'STOP', 'FOG']
        lights_list = [l for l in lights if re.search(l, service_name)]
        if len(lights_list):
            service =  f'REPLACE LIGHT BULBS ({", ".join(lights_list)})'
        else:
            service = service_name
    # BATTERY
    elif cat == 'BATTERIES':
        battery = ['REPLACE', 'CHARGING', 'NEW']
        battery_list = [b for b in battery if re.search(b, service_name)]
        if len(battery_list):
            if 'CHARGING' in battery_list:
                service.append('BATTERY CHARGING')
            if 'REPLACE' in battery_list:
                service.append('BATTERY REPLACEMENT')
            if 'NEW' in battery_list:
                new_battery = re.search('((MEGA FORCE|AC(\s)?DELCO))?(\s)?NS[0-9]{2}', service_name)[0].strip()
                service.append(f"NEW BATTERY {new_battery.upper()}")
            service = ' + '.join(service)
            if 'AC DELCO' in service:
                service = re.sub('AC DELCO', 'ACDELCO', service)
        else:
            service = service_name
    # TIRES  
    elif cat == 'TIRES':
        tires = ['VULCANIZING', 'CHANGE SPARE TIRE', 'BALANCING', 'MOUNTING', 
                 'INSTALLATION', 'ALIGNMENT' , 'NEW']
        tires_list = [t for t in tires if re.search(t, service_name)]
        if len(tires_list):
            if "NEW" in tires_list:
                new_tire = [re.search(tb + ".*[A-Z]*(([0-9]{3})|([0-9]+\.[0-9]*))\/[0-9]{2}\/(R)?[0-9]{2}[A-Z]?", service_name)[0].strip()
                            for tb in tire_brands if re.search(tb, service_name)]
                try:
                    service.append(f"NEW TIRES {new_tire[0]}")
                except:
                    service.append("NEW TIRES")
            if "CHANGE SPARE TIRE" in tires_list:
                service.append("CHANGE SPARE TIRE")
            if "VULCANIZING" in tires_list:
                service.append("VULCANIZING")
            t_list = [t for t in ["BALANCING", "MOUNTING", "INSTALLATION", "ALIGNMENT"] if t in tires_list]
            if len(t_list):
                service.append(f"TIRE {'/'.join(t_list)}")
            else:
                pass
            service = ' + '.join(service)
        else:
            service = service_name
    # AIRCON
    elif cat == "AIRCON":
        aircon_corrections = {'AC CLEANING' : 'AIRCON CLEANING'}
        aircon = ["(FREE)?(\s)?(AC|AIRCON)?(\s)?CLEAN(ING)?", "NO DASHBOARD PULLDOWN", 
                  '(AIRCON|AC) BELT', '((FREE )|(AIR AND )|(CABIN AND ))?(CABIN|AIR) FILTER(\s)?(REPLACEMENT)?',
                  'FREON CHARGING']
        aircon_list = [re.search(a, service_name)[0].strip() for a in aircon if re.search(a, service_name)]
        if len(aircon_list):
            service = ' + '.join(aircon_list)
            for aircon_key in aircon_corrections.keys():
                if re.search(aircon_key, service):
                    service = re.sub(aircon_key, aircon_corrections[aircon_key], service)
                else:
                    continue
        else:
            service = service_name
    
    # BRAKES
    elif cat == "BRAKES":
        brakes_corrections = {'BRAKE(S)?.*((CLEAN(ING|ER)?)|((&|AND)? ADJUST(ER)?)){1}' : 'BRAKES CLEANING AND ADJUSTING'}
        brakes = ['BRAKE(S)?.*((CLEAN(ING|ER)?)|((&|AND)? ADJUST(ER)?)){1}',
                  'BRAKE FLUID TOP[- ]?UP', 'CALIPER (PISTON|KIT (SET)?|BOLT)',
                  'HAND BRAKE CABLE', '(REAR|FRONT)?(\s)?BRAKE PAD(S)?(\s)?(REPLACEMENT)?\s?(.*)?(?=LABOR)',
                  'BRAKE(S)?.*((FLUID)|(FLUSH(ING)?)){1}', 'BRAKE(S)?.*((MASTER)|(CYLINDER|ASSEMBLY)){1}',
                  'BRAKE(S)?.*DRUM REFACING', 'BRAKE(S)?.*SHOE(S)?(REPLACEMENT)?(\s\(.*\))?', 'BRAKE HOSE']
        brakes_list = [re.search(br, service_name)[0].strip() for br in brakes if re.search(br, service_name)]
        if len(brakes_list):
            service = ' + '.join(brakes_list)
            if re.search('LABOR(\s)?(ONLY)?', service_name):
                 service += ' (LABOR)'
                 for brake_key in brakes_corrections.keys():
                     if re.search(brake_key, service):
                         service = re.sub((brake_key, brakes_corrections[brake_key], service))
                     else:
                         continue
            else:
                pass
        else:
            service = service_name
    
    # DIAGNOSIS & INSPECTION
    elif cat == "DIAGNOSIS & INSPECTION":
        diag_corrections = {'CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION' : 'CAR DIAGNOSIS AND VISUAL INSPECTION'}
        diagnosis = ['CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION', 'ENGINE SCANNING']
        diagnosis_list = [re.search(diag, service_name)[0].strip() for diag in diagnosis if re.search(diag, service_name)]
        if len(diagnosis_list):
            service = ' + '.join(diagnosis_list)
            for diag_key in diag_corrections.keys():
                if re.search(diag_key, service):
                    service = re.sub(diag_key, diag_corrections[diag_key], service)
                else:
                    continue
        else:
            service = service_name
    
    # MAINTENANCE
    elif cat == "MAINTENANCE":
        maintain = service_category_dict['MAINTENANCE']
        maintain_list = [re.search(m, service_name)[0].strip() for m in maintain if re.search(m, service_name)]
        if len(maintain_list):
            service = ' + '.join(maintain_list)
            if re.search('REPLACE', service_name):
                service = re.sub('REPLACE', '', service)
                service += ' REPLACEMENT'
            if re.search('LABOR(\s)?(ONLY)?', service_name):
                 service += ' (LABOR)'
        else:
            service = service_name
        
    # PMS
    elif cat == 'PMS':
        service = service_name
    
    elif cat == "DISINFECTION":
        disinfect_corrections = {'DISINFECT' : 'DISINFECTION'}
        #disinfect = ['CAR.*((SANITATION (AND|&)?)|(FOGGING))', 'DISINFECT']
        disinfect = ['SANITATION', 'FOGGING', 'DISINFECT']
        disinfect_list = [re.search(dis, service_name)[0].strip() for dis in disinfect if re.search(dis, service_name)]
        if len(disinfect_list):
            service = 'CAR ' + ' + '.join(disinfect_list)
            for disinfect_key in disinfect_corrections.keys():
                if re.search(disinfect_key, service):
                    service = re.sub(disinfect_key, disinfect_corrections[disinfect_key], service)
                else:
                    continue
        else:
            service = service_name
    
    elif cat == 'MISCELLANEOUS':
        service = service_name

    return service

# def clean_services(service):
#     temp = []
#     for ser in services.keys():
#         if re.search(ser, service.upper()):
#             temp.append(services[ser])
#             if ser == '((ALL IN PMS)|(BIRTHDAY PROMO)|(13TH MONTH PROMO))' or ser == 'ANNUAL.*PMS PACKAGE':
#                 break
#             else:
#                 continue
#         else:
#             continue
#     if len(temp) == 0:
#         temp.append(service.upper())
#     return '+'.join(temp)


def get_prior_services(x, df):
    df_temp = df[(df.date < x['date']) & (df.full_name == x['full_name']) &
                 (df.brand == x['brand']) & (df.model == x['model'])]
    df_temp.loc[:,'service_name'] = df_temp.apply(lambda x: str(x['service_name']), axis=1)
    services = ', '.join(df_temp.service_name.tolist())
    return services

@st.cache_data
def get_data():
    
    '''
    Import data from redash query
    http://app.redash.licagroup.ph/queries/103/source#114
    Perform necessary data preparations
    
    Parameters
    ----------
    None.

    Returns
    -------
    df_data: dataframe
        
    '''
    
    all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/103/results.csv?api_key=QHb7Vxu8oKMyOVhf4bw7YtWRcuQfzvMS6YBSqgeM", 
                           parse_dates = ['date','appointment_date','date_confirmed','date_cancelled'])
    all_data.loc[:,'date'] = pd.to_datetime(all_data.loc[:,'date'])
    # rename columns
    all_data = all_data.rename(columns={'year': 'model_year', 
                                        'name':'status',
                                        'make': 'brand'})
    # filter out non integer results in "model_year"
    all_data = all_data[all_data.loc[:,'model_year'].apply(lambda x: 1 if re.match('\d+', str(x)) else 0) == 1]
    all_data.loc[:, 'model_year'] = all_data.loc[:,'model_year'].apply(lambda x: str(int(x)))
    all_data = all_data[all_data.model_year != '2099']
    
    all_data.loc[:,'brand'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['brand']).upper(), axis=1)
    all_data.loc[:,'model'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['model']).upper(), axis=1)
    all_data.loc[:,'model_year_group'] = all_data.apply(lambda x: model_year_group(int(x['model_year'])), axis=1)
    all_data.loc[:,'year_month'] = all_data.apply(lambda x: str(x['date'].year) + '-' + str(x['date'].month).zfill(2), axis=1)
    
    # remove cancelled transactions
    all_data = all_data[all_data['status']!='Cancelled']
    
    # remove duplicates and fix names
    all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']).title(), axis=1)
    all_data.loc[:, 'model/year'] = all_data.loc[:, 'model'].str.upper() + '/' + all_data.loc[:, 'model_year']
    all_data.loc[:, 'plate_number'] = all_data.plate_number.fillna('').apply(lambda x: re.sub('[- ]', '', re.search('[a-zA-Z]+[0-9]?.*[a-zA-Z]?[0-9]+', x)[0].upper()) 
                                                                             if re.search('[a-zA-Z]+[0-9]?.*[a-zA-Z]?[0-9]+', x) else x)
    all_data.loc[:, 'phone'] = all_data.phone.apply(fix_phone_num)
    all_data.loc[:, 'address'] = all_data.address.apply(fix_address)
    all_data.loc[:, 'mechanic_name'] = all_data.apply(lambda x: fix_name(x['mechanic_name']).title(), axis=1)

        # desired columns
    cols = ['id', 'date', 'email','phone', 'address', 'full_name','brand', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'category', 'service_name', 'sub_total', 'service_fee', 
        'total_cost', 'date_confirmed', 'status', 'status_of_payment','customer_id', 
        'fuel_type', 'transmission', 'plate_number','mileage','model/year',
        'model_year_group', 'year_month', 'lead_source']
    
    # columns used for dropping duplicates
    drop_subset = ['full_name', 'brand', 'model', 'appointment_date', 'date']
    all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
    
    # combine "service name" of entries with same transaction id
    all_data_.loc[:, 'service_category'] = all_data_.loc[:, 'service_name'].apply(lambda x: get_service_category(x))
    all_data_.loc[:, 'service_name'] = all_data_.fillna('').apply(lambda x: get_service_name(x['service_category'], x['service_name'].strip()), axis=1)
    all_data_.loc[:, 'prior_services'] = all_data_.apply(lambda x: get_prior_services(x, all_data_), axis=1)
    #temp = all_data_.fillna('').groupby(['full_name', 'brand', 'model'])['service_name'].apply(lambda x: ', '.join(x).upper()).sort_index(ascending=True).reset_index()
    #temp.loc[:,'last_service'] = all_data_.fillna('').groupby(['full_name', 'brand', 'model'])['service_name'].first().values
    # generate service_name_x and service_name_y
    #df_data = all_data_.merge(temp, left_on=['full_name', 'brand', 'model'], right_on=['full_name', 'brand', 'model'])
    #df_data = pd.concat([all_data_, temp], axis=1)
    df_data = all_data_.copy()
    # convert date to datetime
    df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])
    # df_data = df_data.rename(columns = {'service_name_y' : 'service_name',
    #                                     'brand_x' : 'brand',
    #                                     'model_x' : 'model'})[cols + ['service_category']]
    
    
    
    # converts each row date to a cohort
    df_data.loc[:,'tx_month'] = df_data.apply(lambda row: row['date'].year*100 + row['date'].month, axis=1)
    # get first month & year of first purchase per full_name
    cohorts = df_data.groupby('full_name')[['date', 'tx_month']].min().reset_index()
    cohorts.columns = ['full_name', 'acq_date', 'first_acq']
    # combines cohort and first_cohort
    df_data = df_data.merge(cohorts, on='full_name', how='left')
    
    # remove test entries
    remove_entries = ['mechanigo.ph', 'frig_test', 'sample quotation']
    df_data = df_data[df_data.loc[:,'full_name'].isin(remove_entries) == False]
    
    return df_data


@st.cache_data
def calc_retention_rate(filtered_df, date_range, rate_or_actual):
    '''
    Calculates retention rates for non-first acq cohorts
    
    Parameters
    ----------
    filtered_df : dataframe
        filtered df_data
    date_range: list
        list of int year_month ("%Y%m" or "202201") Note leading zero fill via zfill
    rate_or_actual: str
        string with values of "Rate" or "Actual"
    
    Returns
    -------
    retention_rate_dict: dictionary
        dictionary with date range values as keys and values with list of retention rates per month
    
    '''
    
    retention_rate_dict = {}
    for date_index, date_val in enumerate(date_range):
        unique_users_pool = list(filtered_df[filtered_df['tx_month']==date_val]['full_name'].unique())
        retention_rate = []
        for year_month_index, year_month in enumerate(date_range[date_index:]):
            if year_month_index == 0:
                # get users with more than 1 transaction in same month
                users = filtered_df[filtered_df['tx_month'] == year_month]['full_name'].value_counts()
                curr_unique_users = list(users[users > 1].index)
            else:
                curr_unique_users = list(filtered_df[filtered_df['tx_month']==year_month]['full_name'].unique())
            # curr_unique_users = list((filtered_df[filtered_df['tx_month] == year_month]['full_name'].value_counts() > 1).index)
            # XOR / not intersection
            # new_unique_users = list(set(unique_users_pool) ^ set(curr_unique_users))
            try:
                if rate_or_actual == "Rate":
                    retention_rate.append(round((len(list(set(curr_unique_users) & set(unique_users_pool))))/len(unique_users_pool), 3))
                else:
                    retention_rate.append(len(list(set(curr_unique_users) & set(unique_users_pool))))
            except:
                retention_rate.append(0)
        if rate_or_actual == 'Rate':
            retention_rate.insert(0, 1)
        else:
            retention_rate.insert(0, len(unique_users_pool))
        retention_rate.extend([np.nan]*(len(date_range) - len(retention_rate)))
        retention_rate_dict[date_val] = retention_rate
    return retention_rate_dict
    

def plot_cohort_analysis(df, column_name, value, start_date, rate_or_actual):
    '''
    Calculate and plot return rate for chosen cohort
    '''
    # setup date_range of data and chart
    date_range = pd.date_range(start = datetime(start_date.year, start_date.month, 1), 
                                 end = datetime(datetime.today().year, 
                                                   datetime.today().month, 1)\
                                                 + timedelta(days=30), 
                                 freq='1M')\
                                .strftime('%Y%m').tolist()
    date_range = [int(d) for d in date_range]
    # filter data
    if column_name == "first_acq":
        #cohort_pivot = overall_cohort_analysis(df, start_date, rate_or_actual)
        headers = sorted(list(df['first_acq'].unique()))
        start_month = int(str(start_date.year) + str(start_date.month).zfill(2))
        filtered_df = df[df['first_acq'].isin(list(headers[headers.index(start_month):]))]

    elif column_name == 'mechanic_name':
        customers = df[(df.mechanic_name == value) &
                       (df.date.dt.date >= start_date)]['full_name'].unique()
        filtered_df = df[df.full_name.isin(customers)]
        
    elif column_name == 'lead_source':
        temp_df = df[~(df.lead_source.isnull())]
        if value == 'website_website':
            customers = temp_df[(temp_df.lead_source != 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers)) & (temp_df.lead_source != 'backend')]
        # use first acq and tx_month
        elif value == 'website_backend':
            customers = temp_df[(temp_df.lead_source != 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend') & (temp_df.date.dt.date > temp_df.acq_date)) | 
                                  (temp_df.full_name.isin(customers) & (temp_df.lead_source != 'backend') & (temp_df.date.dt.date == temp_df.acq_date))]
        elif value == 'backend_website':
            customers = temp_df[(temp_df.lead_source == 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers) & (temp_df.lead_source != 'backend') & (temp_df.date.dt.date > temp_df.acq_date)) | 
                                  (temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend') & (temp_df.date.dt.date == temp_df.acq_date))]
        elif value == 'backend_backend':
            customers = temp_df[(temp_df.lead_source == 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend')]
    else:
        filtered_df = df[df[column_name]==value]
        
    # calculate return rate for selected category value
    cohort_dicts = calc_retention_rate(filtered_df, date_range, rate_or_actual)
    cohort_pivot = pd.DataFrame.from_dict(cohort_dicts, orient='index')
    cohort_pivot.columns = range(-1, len(date_range))
    
    fig_dims = (12, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    #ax.set(xlabel='Months After First Purchase', ylabel='First Purchase Cohort', title="Cohort Analysis")
    y_labels = [year_month for year_month in date_range]
    x_labels = np.array(list(range(0, len(y_labels))))-1
    plt.yticks(ticks=range(len(date_range)), labels=y_labels, fontsize=15, rotation=90)
    plt.xticks(x_labels, x_labels, fontsize=15)
    # adjusted scale for colorbar via vmin/vmax
    sns.heatmap(cohort_pivot, annot=True, fmt= '.1%' if rate_or_actual=="Rate" else '.0f' , mask=cohort_pivot.isnull(), 
                square=True, linewidths=.5, cmap=sns.cubehelix_palette(8), annot_kws={"fontsize":11},
                vmin=0, vmax=0.1 if rate_or_actual=="Rate" else np.max(np.max(cohort_pivot.iloc[:,1:])))
    
    plt.xlabel('Months After Acquisition', size=18)
    plt.ylabel('Acquisition Month', size=18)
    plt.title('Cohort Analysis')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    st.dataframe(filtered_df[['date', 'full_name', 'phone', 'lead_source', 'brand', 'model', 'model_year', 'mechanic_name', 'service_name', 'acq_date']])

    
@st.cache_data
def cohort_rfm(df):
    '''
    
    Parameters
    ----------
    df : dataframe
        Prepared customer transaction dataframe

    Returns
    -------
    df_retention : dataframe
        Customer dataframe with extracted RFM features

    '''
    df_retention = df.groupby(['full_name', 'brand', 'model']).agg(first_acq=('date', lambda x: x.min().year*100 + x.min().month),
                                       recency=('date', lambda x: (x.max() - x.min()).days),
                                       frequency=('id', lambda x: len(x) - 1),
                                       total_sales=('total_cost', lambda x: round(np.sum(x), 2)),
                                       avg_sales=('total_cost', lambda x: round(np.mean(x), 2)),
                                       T = ('date', lambda x: (datetime.today()-x.min()).days + 1),
                                       year=('date', lambda x: x.min().year),
                                       month=('date', lambda x: x.min().month),
                                       month_diff=('appointment_date', lambda x: (datetime.today().year - x.max().year)*12 + (datetime.today().month - x.max().month)),
                                       last_mechanic_name=('mechanic_name', 'first'),
                                       last_txn_date = ('appointment_date', lambda x: x.max().date().strftime('%Y/%m/%d'))
                                       )
    df_retention.columns = ['first_acq', 'recency', 'frequency', 'total_sales', 
                         'avg_sales', 'T', 'year', 'month', 'month_diff', 
                         'last_mechanic_name', 'last_txn_date']
    df_retention.loc[:,'ITT'] = df_retention.apply(lambda row: round(get_ratio(row['recency'], row['frequency']), 2), axis=1)
    df_retention.loc[:, 'last_txn'] = df_retention.apply(lambda x: int(x['T'] - x['recency']), axis=1)
    df_retention = df_retention.fillna(0)
    # filter data by returning customers
    df_retention = df_retention[df_retention['avg_sales'] > 0]
    
    return df_retention


def customer_lv_(df_retention):
    '''
    Calculates customer lifetime value

    Parameters
    ----------
    df_retention : dataframe
        Cohort rfm data

    Returns
    -------
    customer_lv : dataframe
        Customer lifetime value and its components

    '''
    
    monthly_clv, avg_sales, purchase_freq, churn = list(), list(), list(), list()

    # calculate monthly customer lifetime value per cohort
    for d in sorted(df_retention['first_acq'].unique()):
      customer_m = df_retention[df_retention['first_acq']==d]
      avg_sales.append(round(np.mean(customer_m['avg_sales']), 2))
      purchase_freq.append(round(np.mean(customer_m['frequency']), 2))
      retention_rate = customer_m[customer_m['frequency']>0].shape[0]/customer_m.shape[0]
      churn.append(round(1-retention_rate,2))
      clv = round((avg_sales[-1]*purchase_freq[-1]/churn[-1]), 2)
      monthly_clv.append(clv)
    
    customer_lv = pd.DataFrame({'first_acq':sorted(df_retention['first_acq'].unique()), 'clv':monthly_clv, 
                                 'avg_sales': avg_sales, 'purchase_freq': purchase_freq,
                                 'churn': churn})
    
    cohorts = sorted(customer_lv['first_acq'].unique())
    data = [customer_lv.clv, customer_lv.churn, customer_lv.avg_sales, 
            customer_lv.purchase_freq]
    y_labels = ['CLV (Php)', 'Churn %', 'Avg Sales (Php)', 'Purchase Freq.']
    
    fig = make_subplots(rows=len(data), cols=1, 
                        shared_xaxes=True, vertical_spacing=0.02)
    
    for i, col in enumerate(data, start=1):
        fig.add_trace(go.Scatter(x=cohorts, y=data[i-1],
                                 line = dict(width=4, dash='dash'),
                                 name= y_labels[i-1]),
                         row=i, col=1)
        fig.update_yaxes(title_text = y_labels[i-1], row=i, col=1)
    fig.update_xaxes(type='category')

    fig.update_layout(title_text = 'Cohort CLV characteristics',
                      height = 1200,
                      width = 800)
    st.plotly_chart(fig)

#@st.cache_data
def bar_plot(df_retention, option = 'Inter-transaction time (ITT)'):
    '''
    Plots inter-transaction time of returning customers

    Parameters
    ----------
    df_retention : dataframe

    Returns
    -------
    ITT plot

    '''
    choice = {'Inter-transaction time (ITT)': 'ITT',
              'Average Sales': 'avg_sales',
              'Days Since Last Transaction': 'last_txn',
              'Active Probability': 'prob_active'}
    
    bins = st.slider('Bins: ', 5, 50, 
                     value=25,
                     step=5)
    
    a = df_retention[df_retention['frequency'] == 1][choice[option]]
    b = df_retention[df_retention['frequency'] > 1][choice[option]]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = a, nbinsx=bins, name='Single Repeat'))
    fig.add_trace(go.Histogram(x = b, nbinsy=bins, name='Multiple Repeat'))
    
    x_lab = {'Inter-transaction time (ITT)': 'Days',
             'Average Sales': 'Amount (Php)',
             'Days Since Last Transaction': 'Days',
             'Active Probability': '%'}
    
    fig.update_layout(barmode='overlay',
                      xaxis_title_text=x_lab[option],
                      yaxis_title_text='Number of customers',
                      title_text='{} by returning customers'.format(option))
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def fit_models(df_retention):
    
    pnbd_filename = 'pnbd.joblib'
    if os.path.exists(pnbd_filename):
        pnbd_dtime = datetime.fromtimestamp(os.path.getmtime(pnbd_filename)).date()
        if (datetime.today().date() - pnbd_dtime).days >= 7:
            pnbd = ParetoNBDFitter(penalizer_coef=0.001)
            pnbd.fit(df_retention['frequency'], df_retention['recency'], df_retention['T'])
        else:
            pnbd = load(pnbd_filename)
    else:
        pnbd = ParetoNBDFitter(penalizer_coef=0.001)
        pnbd.fit(df_retention['frequency'], df_retention['recency'], df_retention['T'])
        dump(pnbd, pnbd_filename)
    
    ggf_filename = 'ggf.joblib'
    if os.path.exists(ggf_filename):
        ggf_dtime = datetime.fromtimestamp(os.path.getmtime(ggf_filename)).date()
        if (datetime.today().date() - ggf_dtime).days >= 7:
            # model to estimate average monetary value of customer transactions
            ggf = GammaGammaFitter(penalizer_coef=0.001)
            # filter df to returning customers
            returning_df_retention = df_retention[df_retention['frequency']>0]
            # fit model
            ggf.fit(returning_df_retention['frequency'], returning_df_retention['avg_sales'])
        else:
            ggf = load(ggf_filename)
    else:
        ggf = GammaGammaFitter(penalizer_coef=0.001)
        returning_df_retention = df_retention[df_retention['frequency']>0]
        ggf.fit(returning_df_retention['frequency'], returning_df_retention['avg_sales'])
        dump(ggf, ggf_filename)
    
    return pnbd, ggf

def plot_prob_active(_pnbd):
    '''
    Plots the active probability matrix for the range of recency and frequency
    
    Parameters
    ----------
    pnbd : model
        Fitted Pareto/NBD model
    '''
    st.title('Active Probability Matrix')
    st.markdown('''
                High recency means customer is most likely still active (long intervals between purchases).\n
                High frequency with low recency means one-time instance of many orders with long hiatus.
                ''')
    fig = plt.figure(figsize=(12,8))
    plot_probability_alive_matrix(_pnbd)
    st.pyplot(fig)

#@st.cache_data
def update_retention(_pnbd, _ggf, t, df_retention):
    # calculate probability of active
    df_retention.loc[:,'prob_active'] = df_retention.apply(lambda x: 
           _pnbd.conditional_probability_alive(x['frequency'], x['recency'], x['T'])*100, 1)
    
    # df_retention.loc[:, 'expected_purchases'] = df_retention.apply(lambda x: 
    #         _pnbd.conditional_expected_number_of_purchases_up_to_time(t, x['frequency'], x['recency'], x['T']),1)
    # df_retention.loc[:, 'prob_1_purchase'] = df_retention.apply(lambda x: 
    #         _pnbd.conditional_probability_of_n_purchases_up_to_time(1, t, x['frequency'], x['recency'], x['T']),1)
    # # predicted average sales per customer
    # df_retention.loc[:, 'pred_avg_sales'] = _ggf.conditional_expected_average_profit(df_retention['frequency'],df_retention['avg_sales'])
    # # clean negative avg sales output from model
    # df_retention.loc[:,'pred_avg_sales'][df_retention.loc[:,'pred_avg_sales'] < 0] = 0
    # # calculated clv for time t
    # df_retention.loc[:,'pred_sales'] = df_retention.apply(lambda x: 
    #         x['expected_purchases'] * x['avg_sales'], axis=1)
    # # round off all columns except cohort
    # round_cols = ['prob_active', 'expected_purchases','prob_1_purchase', 'pred_avg_sales', 'pred_sales', 'last_txn', 'ITT', 'total_sales']
    # df_retention.loc[:, round_cols] = df_retention.loc[:, round_cols].round(3)
    
    return df_retention
        
#@st.cache_data
def search_for_name_retention(name, df_retention):
    '''
    Function to search for customer names in backend data
    '''
    df_retention = df_retention.reset_index()
    # lower to match with names in dataframe
    df_retention.loc[:,'full_name'] = df_retention.apply(lambda x: x['full_name'].lower(), axis=1)
    # search row with name
    names_retention = df_retention[df_retention.apply(lambda x: name.lower() in x['full_name'], axis=1)]
    df_temp_retention = names_retention[['full_name', 'phone', 'address', 'brand', 'model', 'model_year', 'plate_number', 
                                         'service_name', 'prior_services', 'last_mechanic_name', 'frequency', 'ITT', 'last_txn', 
                                         'last_txn_date', 'month_diff', 'prob_active', 'avg_sales', 'total_sales']]
    df_temp_retention.loc[:, 'full_name'] = df_temp_retention.loc[:, 'full_name'].str.title()
    # round off all columns except cohort
    round_cols = ['prob_active','avg_sales', 'last_txn', 'ITT', 'total_sales']
    df_temp_retention.loc[:, round_cols] = df_temp_retention.loc[:, round_cols].round(3)
    #df_temp_retention.loc[:, 'cohort'] = df_temp_retention.loc[:, 'cohort'].apply(int)
    df_temp_retention = df_temp_retention.set_index('full_name')
    return df_temp_retention

def combine_customer_data(df_data, df_retention):
    '''
    Combine dataframes for display/filter-ready use
    '''
    df_temp = df_data.reset_index()[['full_name', 'phone', 'email', 'address', 'brand', 'model', 
                                     'model_year', 'plate_number', 'service_name', 
                                     'prior_services','tx_month', 'lead_source']]\
                                    .drop_duplicates(subset=['full_name', 'brand', 'model'], keep='first')
    
    df_temp_ret = df_retention[['last_mechanic_name', 'frequency', 'ITT', 'last_txn', 'last_txn_date', 'first_acq',  
                                'total_sales', 'avg_sales', 'month_diff',
                                'prob_active']].reset_index()
    
    df_merged = pd.merge(df_temp, df_temp_ret, how='right', left_on=['full_name', 'brand', 'model'], 
                                                            right_on=['full_name', 'brand', 'model']).dropna()
    # Capitalize first letter of each name
    df_merged.loc[:, 'full_name'] = df_merged.loc[:, 'full_name'].str.title()
    return df_merged

def customer_search(df_data, df_retention):
    '''
    Displays retention info of selected customers.

    Parameters
    ----------
    df_data : dataframe
    df_retention : dataframe
    models : list
        list of fitted Pareto/NBD and Gamma Gamma function

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_merged = combine_customer_data(df_data, df_retention)
    
    # table settings
    df_display = df_merged.sort_values(by='full_name')
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column('full_name', headerCheckboxSelection = True)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        autoSizeColumn = 'full_name',
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height=400, 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        
        df_list_retention = [search_for_name_retention(selected[checked_items]['full_name'], df_merged) 
                             for checked_items in range(len(selected))]
        
        df_list_retention = pd.concat(df_list_retention)
        st.dataframe(df_list_retention)          

    else:
        st.write('Click on an entry in the table to display customer data.')
        df_list_retention = pd.DataFrame()
        
    return df_list_retention

def mechanics_utilization(df):
    df_copy = df.copy()
    df_copy.loc[:, 'day_name'] =  df_copy['date'].dt.day_name()
    mechanics_names = sorted(list(df_copy[~df_copy.mechanic_name.isin(['', 
                                                                       'Rapide Dasma', 
                                                                       'Rasi  Head Office',
                                                                       'Lito  Rapide'])]['mechanic_name'].unique()))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #cols = ['total_daily_apps'] + ['mean_apps_' + day for day in day_names]
    df_new = pd.DataFrame({"mechanic_name" : mechanics_names})
    
    # trans/day, trans/weekday, retention_rate, top_services
    def get_mechanics_stats(df, name, day = None):
        if day is not None:
            temp = df[(df.mechanic_name == name) & (df.day_name == day)]
        else:
            temp = df[df.mechanic_name == name]
        try:
            return round(len(temp['day_name'])/temp['date'].nunique(), 3)
        except:
            return 0
    
    for day in day_names:
        df_new.loc[:,'mean_bookings_' + day] = df_new.apply(lambda x: get_mechanics_stats(df_copy, x['mechanic_name'], day), axis=1)
    
    df_new.loc[:, 'mean_weekly_bookings'] = df_new.fillna(0).sum(axis=1,
                                                       skipna = True,
                                                       numeric_only = True)
    df_new.loc[:, 'mean_daily_bookings'] = df_new.apply(lambda x: get_mechanics_stats(df_copy, x['mechanic_name']), axis=1)
    return df_new

def service_distribution(df):
    df_copy = df.copy()
    df_copy.loc[:, 'day_name'] =  df_copy['date'].dt.day_name()
    service_cats = sorted(list(df_copy.service_category.unique()))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #cols = ['total_daily_apps'] + ['mean_apps_' + day for day in day_names]
    df_new = pd.DataFrame({"service_category" : service_cats})
    
    # trans/day, trans/weekday, retention_rate, top_services
    def get_service_stats(df, name, day = None):
        if day is not None:
            temp = df[(df.service_category == name) & (df.day_name == day)]
        else:
            temp = df[df.service_category == name]
        try:
            return round(len(temp['day_name'])/temp['date'].nunique(), 3)
        except:
            return 0
    
    for day in day_names:
        df_new.loc[:,'mean_bookings_' + day] = df_new.apply(lambda x: get_service_stats(df_copy, x['service_category'], day), axis=1)
    
    df_new.loc[:, 'mean_weekly_bookings'] = df_new.sum(axis=1, skipna = True,
                                                       numeric_only = True)
    df_new.loc[:, 'mean_daily_bookings'] = df_new.apply(lambda x: get_service_stats(df_copy, x['service_category']), axis=1)
    return df_new
    
@st.cache_data
def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent recomputation on every rerun.
    return df.to_csv(index = False).encode('utf-8')

def last_update_date():
    return datetime.today().strftime('%Y-%m-%d')

def update():
    st.cache_data.clear()
    st.experimental_rerun()


## =========================== main flow ======================================
if __name__ == '__main__':
    st.title('MechaniGO.ph Customer Retention')
    # import data and preparation
    df_data = get_data()
    
    # calculates cohort rfm data
    df_temp = cohort_rfm(df_data)
    
    # fit pareto/nbd and gamma gamma models
    pnbd, ggf = fit_models(df_temp)
    

    filter_tab, cohort_tab, mechanics_tab, clv_tab = st.tabs(['Customer Filter', 'Cohort Analysis', 'Mechanics', 'CLV'])
    with filter_tab:
        st.markdown("""
                This app searches for the **name** or **phone* you select on the table.\n
                Filter the name/email on the dropdown menu as you hover on the column names. 
                Click on the entry to display data below. 
                """)
                
        # filters
        with st.expander('Customer Data Filters'):
            st.subheader('Last Transaction Date')
            min_txn, max_txn = st.columns(2)
            with min_txn:
                min_txn_date = st.date_input('Min Transaction Date:',
                          min_value = df_data.appointment_date.min(),
                          max_value = datetime.today(),
                          value = df_data.appointment_date.min())
            with max_txn:
                max_txn_date = st.date_input('Max Transaction Date:',
                          min_value = df_data.appointment_date.min(),
                          max_value = datetime.today(),
                          value = datetime.today())
                
            df_temp.loc[:, 'last_txn_date'] = pd.to_datetime(df_temp.last_txn_date)
            df_retention = df_temp[(df_temp.last_txn_date.dt.date >= min_txn_date) & (df_temp.last_txn_date.dt.date <= max_txn_date)]
            df_retention.loc[:, 'last_txn_date'] = df_retention.last_txn_date.apply(lambda x: x.strftime('%Y/%m/%d'))
            df_temp.loc[:, 'last_txn_date'] = df_temp.last_txn_date.apply(lambda x: x.strftime('%Y/%m/%d'))
            
        time = 30
        df_retention = update_retention(pnbd, ggf, time, df_retention)
         
        customer_retention_list = customer_search(df_data, df_retention)
        
        if len(customer_retention_list):
            st.download_button(
                label ="Download customer data",
                data = convert_csv(customer_retention_list),
                file_name = "customer_retention.csv",
                key='download-retention-csv'
                )
            
        # template filters
        # need to check no newer transactions were made
        df_merged = combine_customer_data(df_data, df_retention)
        
        st.header('Customer filter templates')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # 6-months last txn, new customer
            st.markdown('New customer, 6-months last transaction')
            six_month_new = df_merged[(df_merged.frequency == 0) & (df_merged.last_txn >= 180) 
                                      & (df_merged.last_txn <= 210)].sort_values('full_name')\
                                                                    .reset_index()\
                                                                    .drop(columns='index')
            st.download_button('Download',
                                  data = convert_csv(six_month_new),
                                  file_name = 'six_month_new.csv')
            
            st.markdown('Unreturned customers, 6+-months last transaction')
            unreturned = df_merged[df_merged.last_txn >= 180].sort_values('full_name')\
                                                                    .reset_index()\
                                                                    .drop(columns='index')
            
            st.download_button('Download',
                                  data = convert_csv(unreturned),
                                  file_name = 'unreturned_six_months.csv')
        
        with col2:
            # 6-months last txn, 6-months ITT, non new customer
            st.markdown('Returning customer, 6-months last transaction, 6-months ITT')
            six_month_interval = df_merged[(df_merged.frequency > 0) & (df_merged.last_txn >= 150) 
                                      & (df_merged.last_txn <= 180) & (df_merged.ITT >= 150) &
                                      (df_merged.ITT <= 180) & ~(df_merged.full_name.isin(df_merged[df_merged.last_txn < 150]['full_name'].unique().tolist()))].sort_values('full_name')\
                                                              .reset_index()\
                                                              .drop(columns='index')
            st.download_button('Download',
                                  data = convert_csv(six_month_interval),
                                  file_name = 'six_month_interval.csv')
            
            st.markdown('Website Bookings')
            
            website_bookings = df_merged[df_merged.lead_source != 'backend'].sort_values('full_name')\
                                                                            .reset_index()\
                                                                            .drop(columns='index')
            st.download_button('Download',
                                  data = convert_csv(website_bookings),
                                  file_name = 'website_bookings.csv')
            
        with col3:
            # IF follow up
            st.markdown('Inspection Follow-up (IF), 30-45 days last transaction')
            insp_follow_up = df_merged[(df_merged.last_txn >= 30) 
                                      & (df_merged.last_txn <= 45) & 
                                      ~(df_merged.full_name.isin(df_merged[df_merged.last_txn < 30]['full_name'].unique().tolist()))]\
                                                                   .sort_values('full_name').reset_index()\
                                                                   .drop(columns=['index'])
            st.download_button('Download',
                                  data = convert_csv(insp_follow_up),
                                  file_name = 'insp_follow_up.csv')
            
            st.markdown('Backend Bookings')
            
            backend_bookings = df_merged[df_merged.lead_source == 'backend'].sort_values('full_name')\
                                                                            .reset_index()\
                                                                            .drop(columns='index')
            st.download_button('Download',
                                  data = convert_csv(backend_bookings),
                                  file_name = 'backend_bookings.csv')
        
        st.markdown('Compilation')
        merged = pd.concat([six_month_interval,
                            six_month_interval,
                            unreturned,
                            insp_follow_up],
                           axis=0)
        st.download_button('Download',
                              data = convert_csv(merged),
                              file_name = datetime.today().date().strftime('%Y-%b') + '_to_contact.csv')
        
        st.markdown('''
                    Variable meanings: \n
                    \n    
                    - **total/avg_sales**: Total/Average sales of each customer transaction. \n
                    - **ITT**: Inter-transaction time (average time between transactions). \n
                    - **Frequency**: Number of repeat transactions of customer
                    - **last_txn**: Days since last transaction. \n
                    - **prob_active**: Probability that customer will still make a transaction in the future. \n
                    ''')
        
        
        
    # plot_prob_active(pnbd)
    with cohort_tab:
        
        st.header('Customer Segmentation')
        # histogram plots customer info 
        st.write('''
                 This bar plot shows the distribution of single/multiple repeat 
                 transaction(s) based on:
                 ''')
        option = st.selectbox('Variable to show: ', 
                              ('Inter-transaction time (ITT)', 'Average Sales', 
                               'Days Since Last Transaction',
                               'Active Probability'))
        bar_plot(df_retention, option=option)
        
        st.header('Cohort Retention Analysis')
        # plot cohort_retention_chart
        st.write('''This chart shows the retention rate for customers of various cohorts
                 (grouped by first month of transaction). The data shows the portion
                 of customers that return with respect to the starting month.
                 ''')
        #cohort_pivot = cohort_analysis(df_data)
        cat_col, val_col, date_col = st.columns(3)
        with cat_col:
            cat = st.selectbox('Category',
                               options=['first_acq', 'model_year', 'brand',
                                        'service_category', 'mechanic_name', 'lead_source'],
                               index = 0,
                               help = 'Category to filter.')
        with val_col:
            opts = {'first_acq': ['None'],
                    'model_year': list(sorted(df_data['model_year'].unique())),
                    'brand' : list(sorted(df_data['brand'].unique())),
                    'service_category': list(np.unique(list(service_category_dict.keys()))),
                    'mechanic_name': sorted(df_data[df_data['mechanic_name'] != '']['mechanic_name'].unique()),
                    'lead_source': ['website_website', 'backend_website', 'website_backend']}
            
            val = st.selectbox('Category value',
                               options = opts[cat],
                               index = 0,
                               help = 'Value of category to filter.')
        with date_col:
            start_month = st.date_input('Starting month',
                          min_value = df_data.date.min().date(),
                          value = pd.to_datetime('2022-01-01'),
                          help = 'Starting month of data shown.')
        
        rate_or_actual = st.radio("Format",
                 options=["Rate", "Actual"],
                 index = 0,
                 help = 'Format of retention numbers shown.')
        # how to filter new and non-new customers
        plot_cohort_analysis(df_data, cat, val, start_month, rate_or_actual)
    
    with mechanics_tab:
        st.title('Mechanics Data')
        # lifetime transactions
        # monthly/weekly/daily transactions
        # mechanics utilization
        df_ = mechanics_utilization(df_data[~df_data.mechanic_name.isin(['',
                                                                       'Lito Rapide',
                                                                       'Rapide Dasma',
                                                                       'Rasi  Head Office'])])\
            .set_index('mechanic_name')\
            .drop('mean_weekly_bookings', axis=1)
        # sns.heatmap(data = df_.set_index('mechanic_name').drop('mean_weekly_bookings', axis=1), 
        #             annot = True,
        #             vmin = 1,
        #             vmax = 2.5,
        #             cmap = 'flare')
        # plt.show()
        st.header('Mechanics Daily Utilization')
        fig_util = go.Figure(data = go.Heatmap({'z': df_.values,
                                            'x': df_.columns,
                                            'y': df_.index},
                                            zmin = 1,
                                            zmax = 2.5,
                                            ))
        #fig_util = px.imshow(df_.values, text_auto = True)
        fig_util = fig_util.update_traces(text = df_.values,
                                          texttemplate="%{text}")
        fig_util.update_layout(height = 800)
        #plt.show()
        st.plotly_chart(fig_util)
        
        # service distribution
        st.header('Service Category Daily Demand')
        df_service = service_distribution(df_data)\
            .set_index('service_category')\
            .drop('mean_weekly_bookings', axis=1)
        # sns.heatmap(data = df_service.set_index('service_category').drop('mean_weekly_bookings', axis=1), 
        #             annot = True,
        #             vmin = 1,
        #             vmax = 2.5,
        #             cmap = 'flare')
        # plt.show()
        fig_dist = go.Figure(data = go.Heatmap({'z': df_service.values,
                                           'x': df_service.columns,
                                           'y': df_service.index}))
        fig_dist = fig_dist.update_traces(text = df_service.values,
                                          texttemplate="%{text}")
        fig_dist.update_layout(height = 600)
        st.plotly_chart(fig_dist)
    
    with clv_tab:
        st.title('Cohort Lifetime Value')
        # calculates customer rfm data and clv
        st.write('''
                 These plots show the CLV for each cohort and how the trend of each 
                 of its components (frequency, average total sales, churn%) vary.
                 ''')
        clv = customer_lv_(df_retention)
    
    # # initialize session_state.last_update dictionary
    # if 'last_update' not in st.session_state:
    #     st.session_state['last_update'] = {datetime.today().strftime('%Y-%m-%d') : customer_retention_list}
    
    # st.info('Last updated: {}'.format(sorted(st.session_state.last_update.keys())[-1]))
    
    # st.warning('''
    #             If you need to update the lists, the button below will clear the
    #             cache and rerun the app.
    #             ''')
                
    # if st.button('Update'):
      
    #     update()
    
    