#%% slide 4: summary of part 1

my_list = [1, 3.49, 'apple', (1,'orange')]

print('the type of my_list is', type(my_list))
print('the type of my_list[0] is', type(my_list[0]))
print('the type of my_list[1] is', type(my_list[1]))
print('the type of my_list[2] is', type(my_list[2]))
print('the type of my_list[3] is', type(my_list[3]))

#lists are mutable
my_list[0] = 5
print(my_list)

#tuples are immutable
my_tuple = (1, 2)
my_tuple[0] = 2 #this will result in an error
my_list[3][0] = 2 #this will also result in an error

#%% slide 5: summary of part 1

def append_a(test_list):
    '''
    Adds the str 'a' after each element in a user-defined list.
    ex: input = [1, 2]. output = [1, 'a', 2, 'a']
    '''
    new_list = []
    for i in test_list:
        new_list.append(i)
        new_list.append('a')
    return new_list

print(append_a(my_list))

#%% slide 8: loading a csv file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#use pd.read_csv() to read in the data and store in a DataFrame
df = pd.read_csv('C:/Users/Ann/Downloads/grant_and_grant_2014.csv', comment='#')

df.head()

#%% slide 9: Dataframe columns

#slicing a column out of a dataframe by using the column name
df['Average offspring beak depth (mm)']

#%% slide 10: slicing dataframes with booleans

#only want indices with offspring beak depth >= to 11 mm
inds = df['Average offspring beak depth (mm)'] >= 11
inds

#slice out rows we want
df_big_offspring_bd = df.loc[inds]
df_big_offspring_bd

#%% slide 11: indexing with loc

#this will result in an error
df_big_offspring_bd.loc[2]

#this will return the third row
df_big_offspring_bd.iloc[2]

#%% slide 12: renaming columns

rename_dict = {'Average offspring beak depth (mm)' : 'avg_offspring_bd', \
               'Paternal beak depth (mm)' : 'paternal_bd', \
               'Maternal beak depth (mm)' : 'maternal_bd'}

df = df.rename(columns=rename_dict)
df.head()

#%% slide 13: numpy arrays from dataframe columns
    
offspring_bd = df['avg_offspring_bd'].values
paternal_bd = df['paternal_bd'].values
maternal_bd = df['maternal_bd'].values

parental_bd = (paternal_bd + maternal_bd) / 2

#%% slide 14: scatterplots

#plot mean parental beak depth vs mean offspring beak depth
plt.scatter(parental_bd, offspring_bd)
plt.xlabel('parental beak depth (mm)')
plt.ylabel('offpsring beak depth (mm)')
plt.title('parental vs offspring beak depth')
plt.savefig('scatter.png', dpi=500)

#%% slide 15: linear regression with scipy

import scipy.optimize

def linear_fun(x, slope, intercept):
    return slope * x + intercept

#compute the curve fit (guess is unit slope and zero intercept)
popt, covar = scipy.optimize.curve_fit(linear_fun, parental_bd, offspring_bd, 
                                   p0=[1,0])

#%% slide 16: linear regression with scipy

#parse the results
slope, intercept = popt

#print the results
print('slope =', slope)
print('intercept = ', intercept, 'mm')

#%% slide 17: scatterplot with line of best fit

#define range for line
x = np.array([7, 12])
y = linear_fun(x, slope, intercept)

plt.scatter(parental_bd, offspring_bd)
plt.plot(x, y, color='red', linewidth=2)
plt.xlabel('parental beak depth (mm)')
plt.ylabel('offpsring beak depth (mm)')
plt.title('parental vs offspring beak depth')
plt.text(10.5, 8, 'y=%.2fx + %.2f' %(slope, intercept), fontsize=10)
plt.savefig('scatter_w_line.png', dpi=500)

#%% slide 19: making a dataframe

#use a dictionary to make a dataframe
data_dictionary = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data=data_dictionary, index=[49, 48, 1])

df

#returns the second row
df.iloc[1]

#returns the row with the index 1 (third row in this dataframe)
df.loc[1]

#%% slide 20: frog tongue adhesion
    
#read dataset and store in dataframe    
df = pd.read_csv('C:/Users/Ann/Downloads/frog_tongue_adhesion.csv', comment='#')
df.head()

#%% slide 21: data extraction

#extract experiment with index 42
df.loc[42]

#set up boolean slicing
date = df['date'] == '2013_05_27'
trial = df['trial number'] == 3
ID = df['ID'] == 'III'

#slice out the row
df_slice = df.loc[date & trial & ID]

#%% slide 22: computing with dataframes and inserting columns

#add a new columns with impact force in units of newtons
df['impact force (N)'] = df['impact force (mN)'] / 1000

df.head()
#%% slide 23: plotting how impact force correlates with other metrics

plt.scatter(df['impact force (mN)'], df['adhesive force (mN)'])
plt.xlabel('impact force (mN)')
plt.ylabel('adhesive force (mN)')
plt.title('impact force vs adhesive force')
plt.savefig('frog_scatter.png', dpi=500)

#%% slide 24: making subplots

# set figure = fig and subplots = axarr
fig, ax = plt.subplots(4, sharex=True)
plt.xlabel('impact force (mN)')
ax[0].scatter(df['impact force (mN)'], df['impact time (ms)'])
ax[1].scatter(df['impact force (mN)'], df['adhesive force (mN)'])
ax[2].scatter(df['impact force (mN)'], df['total contact area (mm2)'])
ax[3].scatter(df['impact force (mN)'], df['contact pressure (Pa)'])
fig.savefig('frog_subplots.png',dpi=500)

#%% slide 26: subplot parameters

fig, ax = plt.subplots(4, sharex=True, figsize=(7,10))

plt.xlabel('impact force (mN)')

ax[0].scatter(df['impact force (mN)'], df['impact time (ms)'])
ax[0].set_ylabel('impact time (ms)')
ax[0].get_yaxis().set_label_coords(-0.15, 0.5)

ax[1].scatter(df['impact force (mN)'], df['adhesive force (mN)'])
ax[1].set_ylabel('adhesive force (mN)')
ax[1].get_yaxis().set_label_coords(-0.15, 0.5)

ax[2].scatter(df['impact force (mN)'], df['total contact area (mm2)'])
ax[2].set_ylabel('total contact area ($mm^2$)')
ax[2].get_yaxis().set_label_coords(-0.15, 0.5)

ax[3].scatter(df['impact force (mN)'], df['contact pressure (Pa)'])
ax[3].set_ylabel('contact pressure (Pa)')
ax[3].get_yaxis().set_label_coords(-0.15, 0.5)

plt.tight_layout()
fig.savefig('frog_subplots.png',dpi=500)

#%% slide 28: enumerate() as a tool for plotting

subplot_list = ['impact time (ms)', 'adhesive force (mN)', \
                'total contact area (mm2)', 'contact pressure (Pa)']

#enumerate function example  
for i,e in enumerate(subplot_list):
    print(i)
    print(e)
    
#%% slide 29: subplots with enumerate()

#same as before
fig, ax = plt.subplots(4, sharex=True, figsize=(7,10))

plt.xlabel('impact force (mN)')

#instead of typing out each subplot individually, use a for loop
for i,e in enumerate(subplot_list):
    ax[i].scatter(df['impact force (mN)'], df[e])
    ax[i].set_ylabel(e)
    ax[i].get_yaxis().set_label_coords(-0.15, 0.5)

plt.tight_layout()
plt.xlabel('impact force (mN)')
fig.savefig('frog_subplots_looped.png',dpi=500)

#%% slide 30: seaborn: a high level plotting package for matplotlib
import seaborn as sns

#box plot
sns.boxplot(data=df, x='ID', y='impact force (mN)')
plt.savefig('boxplot.png', dpi=500)

#%% slide 31: seaborn: a high level plotting package for matplotlib

#beeswarm plot
sns.swarmplot(data=df, x='ID', y='impact force (mN)')
plt.savefig('beeswarm.png', dpi=500)

#%% slide 32: seaborn: a high level plotting package for matplotlib

#overlay box plot and beeswarm plot
sns.boxplot(data=df, x='ID', y='impact force (mN)', color='lightgray')
sns.swarmplot(data=df, x='ID', y='impact force (mN)')
plt.savefig('boxbeeplot.png', dpi=500)

#%% slide 33: working with gene expression data sets

#read in the scrna-seq dataset. this will take a few seconds
df = pd.read_csv('C:/Users/Ann/Downloads/GBM_raw_gene_counts.csv', sep=' ')
#%% slide 34: non-integer index column

#let's look at the index column
df.index

#df.loc must contain a gene name
df.loc['A2MP1']

#%% slide 35: checking if a gene is in the dataset

#can do this manually 
'TP53' in df.index

def gene_in_df(gene_list):
    """
    This function takes in a list of gene names and 
    returns the names of genes that are not in the dataset.
    """    
    genes_in_df = []
    for i in gene_list:
        if i in df.index:
            genes_in_df.append(i)
    
    print('Gene(s) not in dataset', set(gene_list) - set(genes_in_df))
    return set(gene_list) - set(genes_in_df)

#%% slide 36: checking if a gene is in the dataset

gene_list = ['ETNPPL', 'FGFR3', 'AQP4', 'GJA1', 'AGT', 'MGST1', 'SLC39A12', \
             'SLC25A18', 'GPR98', 'SLCO1C1', 'SDC4', 'GPR37L1', 'ACSBG1', \
             'SFXN5', 'BMPR1B', 'ATP13A4', 'RANBP3L', 'GJB6', 'GFAP', \
             'PRODH', 'SLC4A4', 'TMEM130', 'GABRB2', 'VSNL1', 'GABRA1', \
             'SYNPR', 'THY1', 'CAMK2A', 'MEG3', 'GABRG2', 'CKMT1B', 'CCK', \
             'CHGB', 'SCG2', 'DNM1', 'MAP7D2', 'CELF4', 'CIT', 'UNC80', \
             'NRXN3', 'SCN2A', 'SNAP25']

gene_in_df(gene_list)

#%% slide 37: gene expression correlation using pearsonr 

from scipy.stats import pearsonr

#measure the correlation between TP53 and DERL1
coef, pval = pearsonr(df.loc['TP53'], df.loc['DERL1'])
print('the coefficient is %0.3f' %coef)
print('the p-value is %0.3f' %pval)

#%% slide 38: gene correlation matrix 

gene_list = ['CRIP1', 'S100A8', 'S100A9', 'ANXA1', 'CD14']

#make a new dataframe with your genes of interest
coef_df = df.loc[gene_list]
coef_df.head()

#transpose the dataframe first so genes are in columns
#then calculate pairwise correlation
corr = coef_df.transpose().corr()

#set color palette where low coefs are blue and high coefs are red
#center the color bar at 0
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='RdBu_r', center=0)
plt.savefig('correlation_heatmap.png', dpi=500)

#%% slide 40: fraction of cells expressing a gene

gene_list = ['TMEM119', 'P2RY12', 'GPR34', 'OLFML3', 'SLC2A5', 'SALL1', 'ADORA3']

def gene_frac_df(gene_list):
    '''
    This function takes in a list of gene names and returns the fraction of
    cells that express each gene.
    '''
    #list to store the fraction for each gene
    gene_frac_list = []
    for i in gene_list:
        if i in df.index:
            #if the gene is in the dataset
            #sum the number of cells that have 0 expression
            gene_0 = (df.loc[i]==0).sum()
            #(total number of cells - cells with 0 expression) / total
            gene_frac = (len(df.columns) - gene_0) / len(df.columns)
            
            #add the fraction to the gene_frac_list
            gene_frac_list.append(gene_frac)
        else:
            #if the gene is not in the dataset, append 0 to the list
            gene_frac_list.append(0)
    return gene_frac_list

#%% slide 41: fraction of cells expressing a gene

gene_frac_list = gene_frac_df(gene_list)

#array containing the number of genes
y_pos = np.arange(len(gene_list))

#barh = horizontal bar graph
plt.barh(y_pos, gene_frac_list)
plt.xlim(0,1)
plt.yticks(y_pos, tuple(gene_list))
plt.xlabel('fraction of expressing cells')
plt.savefig('frac_gene.png', facecolor='white', dpi=500)