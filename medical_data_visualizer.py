import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where(df['weight'] / (df['height']/100)**2 > 25, 1, 0)
normalizeCholesterolConditions = [
    (df['cholesterol'] == 1),
    (df['cholesterol'] > 1)
]
normalizeCholesterolValues = [
    0,
    1
]
df['cholesterol'] = np.select(normalizeCholesterolConditions, normalizeCholesterolValues)

normalizeGlucConditions = [
    (df['gluc'] == 1),
    (df['gluc'] > 1)
]
normalizeGlucValues = [
    0,
    1
]
df['gluc'] = np.select(normalizeGlucConditions, normalizeGlucValues)
df_heat2 = df
#df = df.astype('float64')
#print(df.head())
# testing = df[~df['gluc'].isin([0,1])]
# print(testing)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars= ['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], ignore_index=False)
    #print(df_cat)

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    print(df_cat)

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(data=df_cat, x= "variable", col='cardio', y="size", kind="bar", hue="value")
    catplot.set_ylabels("total")

    # Get the figure for the output
    fig = catplot.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    # df_heat = df[df['ap_lo'] <= df['ap_hi']]
    # df_heat = df_heat[df_heat['height'] >= df_heat['height'].quantile(0.025)]
    # df_heat = df_heat[df_heat['height'] <= df_heat['height'].quantile(0.975)]
    # df_heat = df_heat[df_heat['weight'] >= df_heat['weight'].quantile(0.025)]
    # df_heat = df_heat[df_heat['weight'] <= df_heat['weight'].quantile(0.975)]
    df_heat3 = df.loc[(df['ap_lo'] <= df['ap_hi'])&
                (df['height'] >= df['height'].quantile(0.025))&
                (df['height'] <= df['height'].quantile(0.975))&
                (df['weight'] >= df['weight'].quantile(0.025))&
                (df['weight'] <= df['weight'].quantile(0.975))]
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])&
                (df['height'] >= df['height'].quantile(0.025))&
                (df['height'] <= df['height'].quantile(0.975))&
                (df['weight'] >= df['weight'].quantile(0.025))&
                (df['weight'] <= df['weight'].quantile(0.975))]
    print(df_heat.head())
    print(df_heat3.head())
    #df_heat = pd.DataFrame(df_heat)
    # Calculate the correlation matrix
    corr = df_heat.corr()
    print(corr)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))
    #mask = None
    plt.figure(figsize=(16, 6))
    dataplot = sns.heatmap(corr, vmin= -0.16, vmax=0.3, cmap="coolwarm", annot=True, mask=mask, fmt='.1f')
 



    # Set up the matplotlib figure
    fig = dataplot.get_figure()

    # Draw the heatmap with 'sns.heatmap()'



    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
