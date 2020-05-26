import urllib.request
import ssl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import networkx as nx

from matplotlib.offsetbox import  OffsetImage
from matplotlib.patches import Circle, Rectangle, Arc
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



# Grafica de anotaciones por jugador (Grafica de cancha)
def AnotacionesJugador(p_id, season, gametype):

    def draw_court(ax=None, color='black', lw=2, outer_lines=False):
        # If an axes object isn't provided to plot onto, just get current one
        if ax is None:
            ax = plt.gca()

        # Create the various parts of an NBA basketball court

        # Create the basketball hoop
        # Diameter of a hoop is 18" so it has a radius of 9", which is a value
        # 7.5 in our coordinate system
        hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

        # Create backboard
        backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

        # The paint
        # Create the outer box 0f the paint, width=16ft, height=19ft
        outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                              fill=False)
        # Create the inner box of the paint, widt=12ft, height=19ft
        inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                              fill=False)

        # Create free throw top arc
        top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                             linewidth=lw, color=color, fill=False)
        # Create free throw bottom arc
        bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                                linewidth=lw, color=color, linestyle='dashed')
        # Restricted Zone, it is an arc with 4ft radius from center of the hoop
        restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                         color=color)

        # Three point line
        # Create the side 3pt lines, they are 14ft long before they begin to arc
        corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                                   color=color)
        corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
        # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
        # I just played around with the theta values until they lined up with the
        # threes
        three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                        color=color)

        # Center Court
        center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                               linewidth=lw, color=color)
        center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                               linewidth=lw, color=color)

        # List of the court elements to be plotted onto the axes
        court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                          bottom_free_throw, restricted, corner_three_a,
                          corner_three_b, three_arc, center_outer_arc,
                          center_inner_arc]

        if outer_lines:
            # Draw the half court line, baseline and side out bound lines
            outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                    color=color, fill=False)
            court_elements.append(outer_lines)

        # Add the court elements onto the axes
        for element in court_elements:
            ax.add_patch(element)

        return ax

    response = shotchartdetail.ShotChartDetail(
    team_id = 0,
    player_id = p_id,
    season_nullable = season,
    season_type_all_star = gametype)

    content = json.loads(response.get_json())

    # transform contents into dataframe
    results = content['resultSets'][0]
    headers = results['headers']
    rows = results['rowSet']
    shot_df = pd.DataFrame(rows)
    shot_df.columns = headers

    ssl._create_default_https_context = ssl._create_unverified_context
    # we pass in the link to the image as the 1st argument
    pic = urllib.request.urlretrieve(f"http://stats.nba.com/media/players/230x185/{p_id}.png",
                                    f"{p_id}.png")

    # urlretrieve returns a tuple with our image as the first
    # element and imread reads in the image as a
    # mutlidimensional numpy array so matplotlib can plot it
    harden_pic = plt.imread(pic[0])


    # get our colormap for the main kde plot
    # Note we can extract a color from cmap to use for
    # the plots that lie on the side and top axes
    cmap = plt.cm.gist_heat_r

    # n_levels sets the number of contour lines for the main kde plot
    joint_shot_chart = sns.jointplot(shot_df.LOC_X, shot_df.LOC_Y, stat_func=None,
                                     kind='kde', space=0, color=cmap(0.1),
                                     cmap=cmap, n_levels=50)

    joint_shot_chart.fig.set_size_inches(12,11)

    # A joint plot has 3 Axes, the first one called ax_joint,
    # It's the one we want to draw our court onto and adjust some other settings
    ax = joint_shot_chart.ax_joint
    draw_court(ax, color="gray", lw=1)

    # Adjust the axis limits and orientation of the plot in order
    # to plot half court, with the hoop by the top of the plot
    ax.set_xlim(-250,250)
    ax.set_ylim(422.5, -47.5)

    # Get rid of axis labels and tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')

    # Add Data Scource and Author
    ax.text(-250,445,'Data Source: stats.nba.com', fontsize=12)

    # Add Harden's image to the top right
    # First create our OffSetImage by passing in our image
    # and set the zoom level to make the image small enough
    # to fit on our plot
    img = OffsetImage(harden_pic, zoom=0.6)
    # Pass in a tuple of x,y coordinates to set_offset
    # to place the plot where you want, I just played around
    # with the values until I found a spot where I wanted
    # the image to be
    img.set_offset((625,621))
    # add the image
    ax.add_artist(img)

    plt.show()

    
    
def draw_network_all(team,season,game_type,df_involvement,df_pairs,size=(9,8), lw=2.5, seed=None, k=None, lc='black'):
    """
    Función para dibujar gráficas de redes y calcular las medidas de centralidad para sus jugadores

    Función que grafica las gráficas de interacciones entre jugadores en jugadas de anotación. Filtra los datos acorde a los parámetros de equipo, temporada y tipo de juego que recibe. Devuelve además un dataframe con las medidas de cenralidad para los jugadores de ese equipo.
    
    Parameters
    ----------
    team : str
        Clave del equipo de interés (p.ej. 'CLE')
    season : str
        Temporada (p. ej. '2016-2017')
    game_type : str
        Tipo de juego ('regular' o 'playoff')
    df_involvement: dataframe
        Dataframe con jugadores y su colaboración en anotaciones
    df_pairs: dataframe
        Pares de jugadores que intervienen en una jugada de anotación

    Returns
    -------
    dataframe
        Valores de centralidad para el equipo, temporada y tipo de juego indicados en los parámetros

    """
    df_inv = df_involvement
    df_shots = df_pairs
    
    temporada_inv = df_inv['season'] == season
    df_inv = df_inv[temporada_inv]
    tipo_juego_inv = df_inv['gametype'] == game_type
    df_inv = df_inv[tipo_juego_inv]
    equipo_inv = df_inv['team'] == team
    df_inv = df_inv[equipo_inv]
    
    temporada_shot = df_shots['season'] == season
    df_shots = df_shots[temporada_shot]
    tipo_juego_shot = df_shots['gametype'] == game_type
    df_shots = df_shots[tipo_juego_shot]
    equipo_shot = df_shots['team'] == team
    df_shots = df_shots[equipo_shot]
    
    unique_teams = list(df_inv['team'].unique())
    networks = {}
    #Creo nodos
    for i in unique_teams:
        networks[i] = nx.Graph()
        players = list(df_inv.loc[df_inv['team']==i, 'player'])
        for player in players:
            networks[i].add_node(player)
    #Creo aristas
    for row in df_shots.itertuples():
        #Get the team and partnership in question
        team = row.team
        player1 = row.shooter
        player2 = row.assister
        shots = row.count

        #Find the appropriate graph and add the weight
        networks[team].add_edge(player1, player2, weight=shots)
    
    #Get the network and remove isolated nodes
    G = networks[team]
    G.remove_nodes_from(list(nx.isolates(G)))
    
    #Get the team color for the nodes
    #ncolor = df_pairs.loc[df_pairs['team']==team,'TeamColor'].item()
    ncolor = nba_color_codes[team] 
    
    #Calculate the node sizes and the edge weights
    sizes = np.array([df_inv.loc[df_inv['player'] == i,['shot_involvement']].iloc[0].item()
                  for i in G.nodes]) *10 #0 
    #weights = [(G[u][v]['weight']**1.5)*(lw/10) for u,v in G.edges()]
    weights = [(G[u][v]['weight'])*(lw/10) for u,v in G.edges()]
    
    #Draw the plot
    plt.figure(figsize=size)
    nx.draw(G,
            pos=nx.spring_layout(G, k=5.4),
            with_labels=True,
            node_size=sizes,
            node_color=ncolor,
            width=weights,
            font_weight="bold",
            font_color=lc,
            alpha=.9,
            edge_color="grey")
    title = f'{team} - Interdependencia entre los jugadores: Temporada {game_type} {season}'
    plt.title(title)
    
    #Declare columns and create a blank dataframe
    cols = ['Team','Degree','Closeness','Eigen','Betweennes','Pagerank']
    df_centrality = pd.DataFrame(columns=cols)

    #Iterate through the different teams' networks
    for i in unique_teams:
        temp_G = networks[i]

        #For the team, create dictionaries of what we want
        team = {player:i for player in list(temp_G.nodes)}
        degree = dict(nx.degree(temp_G))
        closeness = {i:nx.closeness_centrality(temp_G, i)
                     for i in temp_G.nodes}
        betweenness = nx.betweenness_centrality(temp_G)
        eigen = nx.eigenvector_centrality(temp_G)
        page = nx.pagerank(temp_G, weight='weight')

        #Create a dataframe
        df_temp = pd.DataFrame([team,degree,closeness,eigen,betweenness,page]).T
        df_temp.columns = cols

        #Append it to our centrality dataframe
        df_centrality = pd.concat([df_centrality, df_temp])
        df_centrality = df_centrality.sort_values('Pagerank', ascending=False)
        
    return df_centrality