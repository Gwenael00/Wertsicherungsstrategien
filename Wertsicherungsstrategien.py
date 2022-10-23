# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:19:32 2022

@author: Gwenael00
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql
import datetime
from scipy.stats import norm
import researchpy as rp
from matplotlib.gridspec import GridSpec

np.random.seed(1)
 
#############################################################################################
######################################  Input  ##############################################
#############################################################################################
Zeitintervall = 'm'

MSCI_input_loc = r'F:\Wintersemester 21 22\Bachelorarbeit\Daten\MSCI\MSCI.xlsx'
EONIA_input_loc = r'F:\Wintersemester 21 22\Bachelorarbeit\Daten\EONIA\EONIA.xlsx'

MSCI_World = pd.read_excel(MSCI_input_loc)
EONIA = pd.read_excel(EONIA_input_loc)

#############################################################################################
###################################  Erstelle TimeSeries  ###################################
#############################################################################################

def Create_TimeSeries():
    #Erstelle eine TimerSeries aus MSCI World und EONIA Zeitreihen

    EONIA['EONIA']              = EONIA['EONIA'] / 100 #Eonia in Prozent umwandeln
    MSCI_World['MSCI_World']    = MSCI_World['Value']  
    #Wir benutzen die beobachteten Tage des MSCI World
    TimeSeries                  = pd.merge(MSCI_World[['Date','MSCI_World']],EONIA,on='Date',how='left')
        
    return(TimeSeries)

#Ausführung Funktion zur Erstellung der TimeSeries
TimeSeries = Create_TimeSeries()

#############################################################################################
###################################  Transformation  ########################################
#############################################################################################


def EONIA_Conversion():
    #Die EONIA Rate ist annualisiert, um sie mit unseren Renditen vergleichbar zumachen, müssen wir EONIA wieder auf
    #tägliche bzw. monatliche Werte umwandeln (EONIA wird auf Basis von ACT/360 berechet)
    
    TimeSeries['Jahr']              = pd.DatetimeIndex(TimeSeries['Date']).year
    DaysPerYear                     = []
    
    for date in TimeSeries['Jahr']: #Berechnung der Anzahl der Tage in entsprechenden Jahr
        ActThreeSixty = ql.Actual360().yearFraction(ql.Date(1,1,date),ql.Date(1,1,date+1))
        ActThreeSixty = ActThreeSixty*360
        DaysPerYear.append(ActThreeSixty)
        
    TimeSeries['Dif']               = DaysPerYear
    i                               = 0
            
    #Transformation EONIA in monatliche Werte
    for row in TimeSeries['EONIA']:
        TimeSeries['EONIA'][0] = 0
        TimeSeries.loc[i,'EONIA'] = pow(TimeSeries.loc[i,'EONIA']+1,1/(TimeSeries.loc[i,'Dif']/30))-1 #jährlich -> monatlich
        i = i + 1   
               
    TimeSeries.drop(columns=['Jahr','Dif'],inplace = True)
    TimeSeries.set_index('Date',inplace=True)
        
    TimeSeries['MSCI_World'] = TimeSeries['MSCI_World'].pct_change()
        
    return (TimeSeries)

#Ausführung Transformationen
EONIA_Conversion() 
TimeSeries = TimeSeries[1:]

#############################################################################################
######################################  CPPI & TIPP  ########################################
#############################################################################################

def CPPI(risky_r, safe_r, m=2, start=1000000, floor=0.9, TIPP=False): 

    dates = risky_r.index
    n_schritte = len(dates)
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #S= 439.249*(1+risky_r).cumprod()       ####Für MC
    I = start
    floor_wert = start*floor
    peak = floor_wert

    # erstelle DFs
    V_C = pd.DataFrame().reindex_like(risky_r)
    E_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    Floor_history = pd.DataFrame().reindex_like(risky_r)
    Rf_history = pd.DataFrame().reindex_like(risky_r)
    E_w_history = pd.DataFrame().reindex_like(risky_r)
    Rf_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        if TIPP == True:
            floor_prime = I*floor
            floor_wert = np.maximum(floor_prime,floor_wert)
        cushion = (I - floor_wert)
        E = m*cushion
        E = np.minimum(E,I)
        E = np.maximum(E,0)
        n_e = E//S.iloc[step]
        E = n_e*S.iloc[step]
        Rf = I - E
        E_w = E/I
        Rf_w = 1 - E_w
        I = E*(1+risky_r.iloc[step]) + Rf*(1+safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        E_history.iloc[step] = E
        V_C.iloc[step] = I
        Floor_history.iloc[step] = floor_wert
        Rf_history.iloc[step] = Rf
        Rf_w_history.iloc[step] = Rf_w
        E_w_history.iloc[step] = E_w
    result = {
        "V_C": V_C, 
        "Cushion": cushion_history,
        "E": E_history,
        "MSCI_World_r":risky_r,
        "EONIA_r": safe_r,
        "Floor": Floor_history,
        "S": S,
        "Rf":Rf_history,
        "E_w":E_w_history,
        "Rf_w":Rf_w_history
    }
    return result

def CPPI_TIPP(TimeSeries, m=2,floor=0.9,start=1000000):  ######Damit in der Auswertung CPPI und TIPP nicht einzeln ausgeführt werden müssen
    
    start = start
    m = m
    floor = floor
    risky_r = TimeSeries[['MSCI_World']]
    safe_r = TimeSeries[['EONIA']]
    risky_r[0] = risky_r['MSCI_World']
    safe_r[0] = safe_r['EONIA']
    risky_r = risky_r.reset_index().drop(["Date",'MSCI_World'], axis=1)
    safe_r = safe_r.reset_index().drop(["Date",'EONIA'], axis=1)

    
    Backtest_CPPI = CPPI(risky_r = risky_r,safe_r=safe_r,TIPP=False,m=m,floor=floor,start=start)
    Backtest_TIPP = CPPI(risky_r = risky_r,safe_r=safe_r,TIPP=True,m=m,floor=floor,start=start)
    
    return (Backtest_CPPI,Backtest_TIPP)

#############################################################################################
#######################################   Simulation ########################################
#############################################################################################

def MC_Simulation(asset):

    asset = asset.astype(float)
    #Parameter der diskreten Rendite
    mu_asset = np.mean(asset)
    sigma_asset = np.std(asset, ddof=1)
    #tramsformiere in LOG-Normalverteilte ZV
    v = np.sqrt(np.log(1+(pow(sigma_asset,2)/pow(1+mu_asset,2))))
    m = np.log(1+mu_asset) - 0.5*pow(v,2)

    n = 200000 #Anzahl Pfade

    MC = list(range(1,n+1)) #DataFrame in Abh. von Anzahl der Pfade
    MC = pd.DataFrame(MC)       
    
    MC_Zeit = 24 #Zeitspanne in Monaten
    
    for i in range(MC_Zeit): #generiere Standardnormalverteilte Zufallsvariablen
        MC[i] = np.random.normal(0, 1, n)
    
    MC = np.exp(m + v*MC) #Errechne lognormalverteilte Renditen aus Zufallszahlen

    MC_cum = MC.transpose()
    MC_cum = MC_cum-1
    
    return (MC_cum)

MC_R_MSCI_World = MC_Simulation(TimeSeries['MSCI_World']) ####Simulierte Renditen MSCI World
MC_EONIA = MC_Simulation(TimeSeries["EONIA"]) #Simulierte EONIA Werte

asset = TimeSeries['MSCI_World'].astype(float)
mu_asset = np.mean(asset)
sigma_asset = np.std(asset, ddof=1)
v = np.sqrt(np.log(1+(pow(sigma_asset,2)/pow(1+mu_asset,2))))
m = np.log(1+mu_asset) - 0.5*pow(v,2)

#############################################################################################
#####################################  BS Funktionen ########################################
#############################################################################################

N = norm.cdf

def BS_CALL(S, X, T, r, sigma): #BS Formel für europäische Call Option
    d1 = (np.log(S/X) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - X * np.exp(-r*T)* N(d2)

def BS_PUT(S, X, T, r, sigma): #BS Formel für europäische Put Option
    d1 = (np.log(S/X) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return X*np.exp(-r*T)*N(-d2) - S*N(-d1)

#############################################################################################
###################################  Covered Short Call #####################################
#############################################################################################

sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World']),ddof=1)*np.sqrt(12) ###Für MC
log_rf = np.log(1+np.mean(TimeSeries['EONIA'][1:]))*12 ####Für MC


def CVSC(risky_r,safe_r,I=1000000,A=8000,K=1):
        
    dates = risky_r.index
    n_schritte = len(dates)
    SP= (I-A)/(len(risky_r)-1)
    #S = 439.249*(1+risky_r).cumprod() ###Für MC
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #P_start = 439.249 ###Für MC
    P_start = MSCI_World['Value'].iloc[276-n_schritte]###Für Backtest
    B_Prime_history = pd.DataFrame().reindex_like(risky_r)
    V_C = pd.DataFrame().reindex_like(risky_r)
    n_history = pd.DataFrame().reindex_like(risky_r)
    Call_Price_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World'][:12+step]),ddof=1)*np.sqrt(12) ##Für Backtest
        log_rf = np.log(1+np.mean(TimeSeries['EONIA'][:12+step]))*12 ###Backtest
        if step== 0:
            Call_Price = BS_CALL(P_start,K*P_start,1/12,log_rf,sigma_asset)
            n = A//(P_start - Call_Price)
            B_Prime = A - n*(P_start - Call_Price)
            I = n*(np.minimum(S.iloc[step],K*P_start)) + B_Prime*(1+safe_r.iloc[step])
        
        else:
            Call_Price = BS_CALL(S.iloc[step-1],K*S.iloc[step-1],1/12,log_rf,sigma_asset)
            n = (I+SP)//(S.iloc[step-1] - Call_Price)
            B_Prime = (I+SP) - n*(S.iloc[step-1] - Call_Price)
            I = n*(np.minimum(S.iloc[step],K*S.iloc[step-1])) + B_Prime*(1+safe_r.iloc[step])
        
        B_Prime_history.iloc[step] = B_Prime
        V_C.iloc[step] = I+SP*(len(risky_r)-1-step)
        n_history.iloc[step] = n
        Call_Price_history.iloc[step] = Call_Price
    
    result = {
        "V_C": V_C, 
        "B_Prime": B_Prime_history,
        "n": n_history,
        "Call_Price": Call_Price_history,
        }
    return result      

def PP(risky_r,safe_r,I=1000000,A=8000,K=1): 
    
    dates = risky_r.index
    n_schritte = len(dates)
    SP= (I-A)/(len(risky_r)-1)
    #S = 439.249*(1+risky_r).cumprod() ###Für MC
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #P_start = 439.249 ###Für MC
    P_start = MSCI_World['Value'].iloc[276-n_schritte]###Für Backtest
    B_Prime_history = pd.DataFrame().reindex_like(risky_r)
    V_C = pd.DataFrame().reindex_like(risky_r)
    n_history = pd.DataFrame().reindex_like(risky_r)
    Put_Price_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World'][:12+step]),ddof=1)*np.sqrt(12) ##Für Backtest
        log_rf = np.log(1+np.mean(TimeSeries['EONIA'][:12+step]))*12 ### Für Backtest
        if step== 0:
            Put_Price = BS_PUT(P_start,K*P_start,1/12,log_rf,sigma_asset)
            n = A//(P_start + Put_Price)
            B_Prime = A - n*(P_start + Put_Price)
            I = n*(np.maximum(S.iloc[step],K*P_start)) + B_Prime*(1+safe_r.iloc[step])
        
        else:
            Put_Price = BS_PUT(S.iloc[step-1],K*S.iloc[step-1],1/12,log_rf,sigma_asset)
            n = (I+SP)//(S.iloc[step-1] + Put_Price)
            B_Prime = (I+SP) - n*(S.iloc[step-1] + Put_Price)
            I = n*(np.maximum(S.iloc[step],K*S.iloc[step-1])) + B_Prime*(1+safe_r.iloc[step])
        
        B_Prime_history.iloc[step] = B_Prime
        V_C.iloc[step] = I + SP*(len(risky_r)-1-step)
        n_history.iloc[step] = n
        Put_Price_history.iloc[step] = Put_Price
    
    result = {
        "V_C": V_C, 
        "B_Prime": B_Prime_history,
        "n": n_history,
        "Put_Price": Put_Price_history,
        }
    return result       
