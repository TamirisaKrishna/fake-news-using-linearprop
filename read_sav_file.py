import pyreadstat
import pandas as pd
#df, meta = pyreadstat.read_sav('C:/Users/TKC/fake-news/final_model.sav')
#df.head()
df = pd.read_spss('C:/Users/TKC/fake-news/final_model.sav')
df.tail()