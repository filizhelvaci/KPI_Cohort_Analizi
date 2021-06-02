############################################
# KPI & COHORT ANALIZI: RETENTION RATE
############################################

# 3 ADIMDA RETENTION RATE KPI'NIN COHORT ANALIZINE SOKULMASI

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from operator import attrgetter
import matplotlib.colors as mcolors

df_ = pd.read_excel('DataSet/online_retail.xlsx')
df = df_.copy()

####################################
# 1. Veri ön işleme
####################################
# CustomerID si na olan satırları uçuruyoruz
df.dropna(subset=['CustomerID'], inplace=True)

# Çoklama kayıtları düşürüyoruz
df = df[['CustomerID', 'InvoiceNo', 'InvoiceDate']].drop_duplicates()

####################################
# 2. Retention matrisinin oluşturulması
####################################

# Her bir müşteri için eşsiz sipariş sayısının hesaplanması
n_orders = df.groupby(['CustomerID'])['InvoiceNo'].nunique()

# Tüm veri setinde bir kereden fazla sipariş veren müşteri oranı.
orders_perc = np.sum(n_orders > 1) / df['CustomerID'].nunique()

# Sipariş aylarının yakalanması.
df['order_month'] = df['InvoiceDate'].dt.to_period('M')

# Cohort değişkeninin oluşturulması.
df['cohort'] = df.groupby('CustomerID')['InvoiceDate'] \
    .transform('min') \
    .dt.to_period('M')

# Aylık müşteri sayılarını çıkarılması.
df_cohort = df.groupby(['cohort', 'order_month']) \
    .agg(n_customers=('CustomerID', 'nunique')) \
    .reset_index(drop=False)


# Periyod numarasının çıkarılması
(df_cohort.order_month - df_cohort.cohort).head()

df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))

# Cohort_pivot'un oluşturulması
cohort_pivot = df_cohort.pivot_table(index='cohort',
                                     columns='period_number',
                                     values='n_customers')

cohort_size = cohort_pivot.iloc[:, 0]

# Retention_matrix'in oluşturulması
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
retention_matrix

####################################
# 3. Retention matrisinin ısı haritası ile görselleştirilmesi
####################################
sns.axes_style("white")
fig, ax = plt.subplots(1, 2,
                       figsize=(12, 8),
                       sharey=True,  # y eksenini paylas
                       gridspec_kw={'width_ratios': [1, 11]}
                       # to create the grid the subplots are placed on
                       )

# retention matrix
sns.heatmap(retention_matrix,
            annot=True,
            fmt='.0%',  # grafikteki ifadelerin yüzdelik gösterimi
            cmap='RdYlGn',  # colormap
            ax=ax[1])  # subplot'taki grafikleri seçmek

ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
ax[1].set(xlabel='# of periods', ylabel='')


# cohort size
cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
white_cmap = mcolors.ListedColormap(['white'])
sns.heatmap(cohort_size_df,
            annot=True,
            cbar=False,  # ikinci grafik için cbar istemiyoruz (sağ taraftaki renkli ölçeklendirme)
            fmt='g',
            cmap=white_cmap,
            ax=ax[0])
fig.tight_layout()
plt.show()