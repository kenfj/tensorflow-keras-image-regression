# %%
"""explore data photo-video_sb.csv"""
import decimal
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df2 = pd.read_csv('./data/photo-video_sb.csv')

# %%
print(df2.shape)
df2.head()

# %%
df2.index.size, df2.id.nunique()

# %%
df2 = df2.drop_duplicates()

# %%
df2.index.size, df2.id.nunique()

# %%
df2.groupby("id")["name"].value_counts()

# %%
df2.loc[df2.name == "Google"]

# %%
max_rating_count_ids = df2.groupby(
    ["id"], sort=False, as_index=False
)["rating_count"].max()

print(max_rating_count_ids)

# %%
merged = df2.merge(max_rating_count_ids, on=["id", "rating_count"])
merged.shape

# %%
merged.id.value_counts()

# %%
merged[merged.id.isin([430437503, 869117407])]

# %%
df3 = merged.drop_duplicates(subset='id', keep="last")
df3.shape

# %%
df3.head()

# %%
# calculate weighted average rating

# need to copy to avoid SettingWithCopyWarning
# https://stackoverflow.com/a/49603010
df3 = df3.copy()

df3.rating_count_list = df3.rating_count_list.apply(json.loads)

# %%
df3["rating_value2"] = df3.rating_count_list.apply(
    lambda x:
    (x[0] + x[1]*2 + x[2]*3 + x[3]*4 + x[4]*5) /
    (sum(x) - sys.float_info.epsilon)
)

# %%
df3.head()

# %%


def rounder(num, prec):
    """
    to avoid using pandas __round__
    round-toward-even is the default as it is in IEEE 754.
    https://stackoverflow.com/questions/10825926/python-3-x-rounding-behavior
    """
    return float(decimal.Decimal(num).quantize(
        decimal.Decimal(str(prec)), rounding=decimal.ROUND_HALF_UP
    ))


# verify the weight average is accurate
df3["rating_value3"] = df3.rating_value2.apply(lambda x: rounder(x, 0.1))
df3.loc[(df3.rating_value3 != df3.rating_value)]

# %%
df3.sort_values("rating_value", ascending=False)

# %%
sns.pairplot(df3, vars=["rating_value2", "rating_count",
             "chart_position"], diag_kind='kde')

# %%
plt.hist(df3.rating_value2, bins=10)

# %% [markdown]
# # Create New Dataframe

# %%
df = df3[["id", "rating_value2"]].copy()
df.reset_index(drop=True, inplace=True)
df.rename({"rating_value2": "rating"}, axis=1, inplace=True)
df.shape

# %%
df.tail()

# %%
df.rating.describe()

# %%
