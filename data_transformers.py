from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

class SMILESTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    ''

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    descs = []
    for sm in X:
      mol = Chem.MolFromSmiles(sm)
      desc = self.getMolDescriptors(mol)
      descs.append(desc)
    
    desc = pd.DataFrame(descs)
    desc = desc.fillna(desc.mean())
        
    return(desc)

  def getMolDescriptors(self, mol, missingVal=None):
    ##https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
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