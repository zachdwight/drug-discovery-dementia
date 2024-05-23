import pandas as pd
from chembl_webresource_client.new_client import new_client #https://www.ebi.ac.uk/chembl/#
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

#import list of potential molecules
data = pd.read_csv('dementia_related_molecules.csv')

str_list = data.CHEMBL_ID.tolist()

molecule = new_client.molecule
mols = molecule.filter(molecule_chembl_id__in=str_list)
targets = pd.DataFrame.from_dict(mols)


#column_names = list(targets.columns.values)
#print(column_names)

#to simplify the example, we select one molecule from our list of targets for evaluation
selected_target = targets.molecule_chembl_id[5]
print(selected_target)

activity = new_client.activity
res = activity.filter(molecule_chembl_id = selected_target).filter(standard_type = "IC50")


df = pd.DataFrame.from_dict(res) 
     
df.standard_type.unique()
df.to_csv('bioactivity_data_raw_dementia.csv', index=False)


#handle the missing values

df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]


df2.to_csv('bioactivity_data_raw_dementia2.csv', index=False)

#trim down the data set
selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2[selection]

df3.to_csv('bioactivity_data_raw_dementia3.csv', index=False)

bioactivity_class = []
for i in df3.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("intermediate")


df3['bioactivity_class'] = bioactivity_class

df4=df3

#save our data
df4.to_csv('bioactivity_data_preprocessed_dementia.csv', index=False)

########################## PART II



# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors
     

df_lipinski = lipinski(df4.canonical_smiles)
     
df4["MW"] = df_lipinski["MW"]
df4["LogP"] = df_lipinski["LogP"]
df4["NumHDonors"] = df_lipinski["NumHDonors"]
df4["NumHAcceptors"] = df_lipinski["NumHAcceptors"]

df_2class = df4[df4.bioactivity_class != 'intermediate']

df_2class.to_csv('bioactivity_data_forCharts_dementia.csv', index=False)

import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt


#CLASSES FREQUENCY CHART
plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='black')

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

plt.savefig('plot_bioactivity_class.pdf')

#scatter plot LogP
plt.figure(figsize=(5.5, 5.5))

sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='standard_value', edgecolor='black', alpha=0.7)

plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('plot_MW_vs_LogP.pdf')
