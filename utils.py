
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import Draw

from dash import dcc, html, dash_table
import glob
import os
import umap.umap_ as umap
import joblib

from data_transformers import *



#https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html
def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList: #use descriptors_list for choosing descriptors or Descriptors._descList() for all of them
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def RFR_to_dataframe(full_data, train_data, test_data, property_truth, pred_name, error_name):
    RFR_df = pd.DataFrame()

    train_X = train_data['SMILES'].copy()
    val_X = test_data['SMILES'].copy()
    train_y = train_data[property_truth].copy()
    val_y = test_data[property_truth].copy()

    model = Pipeline(steps=[
                       ('descriptors', SMILESTransformer()),    
                       ('rf', RandomForestRegressor())
                ])


    print("training...")
    model.fit(train_X, train_y)
    print("done training")
    predictions = model.predict(val_X)
    RFR_pred = pd.DataFrame(predictions).rename(columns={0:pred_name})
    val_y.index = range(len(val_y))
    RFR_error = pd.DataFrame(abs(RFR_pred[pred_name] - val_y)).rename(columns={0:error_name})
    RFR_pred = RFR_pred.reset_index().merge(RFR_error.reset_index(),on='index')
    RFR_pred.index = range(len(RFR_pred))
    RFR_df = pd.concat([RFR_df,RFR_pred],axis=0)

    RFR_df.index = test_data.index
    RFR_pred_final = pd.concat([test_data.loc[:,['SMILES',property_truth]],RFR_df.iloc[:,1:]],axis=1)

    RFR_pred_final = RFR_pred_final.merge(full_data.reset_index()[['index','SMILES']],on='SMILES').set_index('index')

    return RFR_pred_final


def get_div1(property_list, initial_similarity_table_data):

    output = []
    for prop in property_list:
        output += [html.Label(prop,style={'display':'inline-block','width':'15vh','vertical-align':'middle','text-align':'right'}),
                   dcc.Dropdown(['','<','>'],'',id={"type":'re-dropdown','index':f're-dropdown-{prop}'},
                                placeholder='operator',style={'display':'inline-block','width':'5vh','vertical-align':'middle'}),
                   dcc.Input(placeholder='value or item',
                             id={'type': 're-input', 'index':f're-input-{prop}'},
                             style={'display':'inline-block','width':'5vh','vertical-align':'middle'})]
        


    output += [html.Br(),
               dash_table.DataTable(data=initial_similarity_table_data.to_dict('records'), sort_action="native", page_action="native", page_size=16, id='re-similarity-descriptor-table',
                                                        columns=[{'name': i, 'id': i} for i in initial_similarity_table_data.columns if i != 'id'],
                                                        style_cell={'overflow': 'hidden','textOverflow': 'ellipsis','maxWidth': 0,},
                                                        tooltip_data=[{column: {'value': str(value), 'type': 'markdown'}
                                                                for column, value in row.items()
                                                            } for row in initial_similarity_table_data.to_dict('records')],
                                                        tooltip_duration=None)]
    

    return(output)


def load_data():

    print('Reading data...')
    fns = glob.glob('property_data/*/*/*')

    property_list = []
    all_data = None
    for fn in fns:

        path = fn.split('/')
        property = path[1]
        dataset = f'{path[2]}'
        model = path[3]

        dfs = []
        for split in glob.glob(f'{fn}/*.csv'):

            df = pd.read_csv(split)
            split = split.split('/')[-1].split('.')[0]
            

            df = df.dropna(subset=['SMILES'])
            if 'name' in df.columns:
                df = df.drop(columns=['name'])

            for col in df.columns:
                if 'Unnamed' in col:
                    df = df.drop(columns = [col])

            df[f'{property}_{dataset}_{model}_data_state'] = split
            df[f'{property}_{dataset}_{model}_error'] = (df['pred'] - df['label']).abs()

            df = df.rename(columns = {'pred':f'{property}_{dataset}_{model}_pred',
                                    'label':f'{property}_truth'})

            sms = []
            formulas = []
            for sm in df['SMILES'].values:
                m = Chem.MolFromSmiles(sm)
                formula = CalcMolFormula(m)
                sm_standard = Chem.MolToSmiles(m)
                sms.append(sm_standard)
                formulas.append(formula)

            df['SMILES'] = sms
            df['formula'] = formulas


            dfs.append(df)
        if len(dfs) == 0:
            continue
        df = pd.concat(dfs)

        if property not in property_list:
            property_list.append(property)

        if all_data is None:
            all_data = df.copy()
        else:
            common_cols = list(set(all_data.columns).intersection(set(df.columns)))
            all_data = all_data.merge(df,on=common_cols, how='outer')
        
    all_data['id'] = range(len(all_data))
    
    return(all_data, property_list)


def generate_images(all_data):
    
    if not os.path.exists("./image_assets"):
        os.makedirs('./image_assets')

    if not os.path.exists("./image_assets/molecule_structures"):
        os.makedirs('./image_assets/molecule_structures')

    print('Generating images..')
    for i in range(len(all_data)):
        if not os.path.exists(r'image_assets/molecule_structures/' + str(all_data.loc[i,'id']) + '-' + all_data.loc[i,'formula'] + ".png"):
            molecule = Chem.MolFromSmiles(all_data.loc[i,'SMILES'])
            img = Draw.MolToImage(molecule)
            img.save(r'image_assets/molecule_structures/' + str(all_data.loc[i,'id']) + '-' + all_data.loc[i,'formula'] + ".png")
            
def process_data(all_data):
    
    print('Starting descriptors...')

    mols = [Chem.MolFromSmiles(smi) for smi in all_data['SMILES']]
    desc = pd.DataFrame([getMolDescriptors(m) for m in mols])

    mol_descriptors = list(desc.columns)

    for col in desc.columns:
        nan_frac = desc[col].isna().sum()/float(len(desc))

        if nan_frac > 0.25:
            desc = desc.drop(columns=[col])
        elif nan_frac > 0:
            mean_value=desc[col].mean() 
            desc[col] = desc[col].fillna(mean_value)

    all_data = pd.concat([all_data,desc],axis=1)

    sc = StandardScaler()

    umap_model = umap.UMAP()
    embedding = umap_model.fit_transform(sc.fit_transform(desc.values))

    um2d = pd.DataFrame(embedding,columns=['dim1','dim2'])

    all_data = pd.concat([all_data,um2d],axis=1)
    
    return(all_data, mol_descriptors, umap_model)


def load_models(all_data):
    
    saved_models = {}

    fns = glob.glob('property_data/*/*/*/model/*.joblib')
    for fn in fns:
        property = fn.split('/')[1]
        dataset = fn.split('/')[2]
        model = fn.split('/')[3]
        model_id = f'{property}_{dataset}_{model}'

        model = joblib.load(fn)
        saved_models[model_id] = model

        preds = model.predict(all_data.SMILES)

        all_data[f'{model_id}_pred'] = preds 
        
    return(all_data, saved_models)