import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from nba_api.stats.endpoints import leagueleaders
from nba_api.stats.endpoints import shotchartdetail

def df_api_lideres(temporada,tipo):
    response = leagueleaders.LeagueLeaders(
    season=temporada,
    season_type_all_star=tipo
    )
    df = response.get_data_frames()[0]
    #df.head(10)
    df1 = df[['PLAYER_ID','RANK','PLAYER','TEAM','AST','PTS']]
    df1=df1.head(20)
    return df1
def grafica_confeti(df):
    df1 = df[['PLAYER_ID','RANK','PLAYER','TEAM','PTS','AST']]
    df1 =df1.head(15)
    plt.rcParams["figure.facecolor"] = 'white'
    plt.figure(figsize=(11,9))
    Y=df1['AST']
    X=df1['PTS']
    g=sns.scatterplot(x=X, y=Y, hue="TEAM", data=df1,s=500)
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(point['val']))
    label_point(X, Y, df1.PLAYER, g)  
    # Customize the axes and title
    g.set_title("Puntos y Asistencias Playoffs 2016-17",fontsize= 20)
    g.set_xlabel("Puntos Totales",fontsize= 13)
    g.set_ylabel("Asistencias Totales",fontsize= 13)
    plt.legend(frameon=False,loc='upper left')
    sns.despine(left=False, bottom=False)
    
def grafica_eff(df):
    df=df.head(10)
    plt.figure(figsize=(11,9))
    x=sns.barplot(x=df.EFF, y=df.PLAYER, data=df)
    x.set_title("Eficiencia Jugadores Playoffs 2016-17",fontsize= 15)
    x.set_xlabel("Eficiencia(EFF)",fontsize= 12)
    x.set_ylabel("Jugador",fontsize= 12)
    sns.despine(left=False, bottom=False)
    
def grafica_FG(df):
    df2 = df[['PLAYER_ID','RANK','PLAYER','TEAM','FG_PCT','FG3_PCT']]
    df2 =df2.head(15)
    Y1=df2['FG3_PCT']
    X1=df2['FG_PCT']
    plt.figure(figsize=(15,10))
    g1=sns.scatterplot(x=X1, y=Y1,
                  hue="TEAM",
                  data=df2,
                  s=300,
                  )
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.001, point['y'], str(point['val']))
    label_point(X1, Y1, df2.PLAYER, g1)  
    g1.set_title("FG% y 3FG% 2016-17",fontsize= 20)
    g1.set_xlabel("FG%",fontsize= 13)
    g1.set_ylabel("3FG%",fontsize= 13)
    plt.legend(frameon=False,loc='upper left')