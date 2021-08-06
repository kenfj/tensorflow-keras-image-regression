# %%
"""explore data photo-video.csv"""
from datetime import datetime
import pandas as pd
import seaborn as sns

# %%
df1 = pd.read_csv('./data/photo-video.csv', index_col="id")

# %%
print(df1.shape)
df1.head()

# %%
df1.loc[df1.index.isin([1085652055, 712908978])]

# %%
# fix Albanian date 11 Nis 2016 (1085652055)
df1.at[df1.date_published == "11 Nis 2016", "date_published"] = "2016-04-11"
# fix Dutch date 8. Okt. 2013 (712908978)
df1.at[df1.date_published == "8. Okt. 2013", "date_published"] = "2013-10-08"

# %%
df1.loc[df1.index.isin([1085652055, 712908978])]

# %%
df1["date_published_years"] = (
    datetime.now() - pd.to_datetime(df1.date_published)
).apply(lambda x: float(x.days)/365)

# %%
df1.head()

# %%
sns.pairplot(df1, diag_kws={'bins': 10})

# %%
sns.pairplot(df1, diag_kind='kde')
