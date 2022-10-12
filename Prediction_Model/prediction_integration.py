import pandas as pd 
import numpy as np
from IPython.display import display



### SKU Prediction Aggregation data 
### Nexgard
nexgard_sku_lst = [str(i) for i in range(142828, 142836)]
print(nexgard_sku_lst)

sku = "142828"
nexgard_data = pd.read_csv(f"C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Nexgard\\{sku}.csv")
nexgard_data.rename(columns={sku: "Gross_Revenue"}, inplace=True)
nexgard_data['Date'] = nexgard_data['Date'].astype('datetime64')
nexgard_data['Year'] = nexgard_data['Date'].dt.year
nexgard_data['Month'] = nexgard_data['Date'].dt.month
nexgard_data['SKU'] = sku
# print(data.head(50))


for sku in nexgard_sku_lst[1:]:
    data_ = pd.read_csv(f"C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Nexgard\\{sku}.csv")
    data_.rename(columns={sku: "Gross_Revenue"}, inplace=True)
    data_['Date'] = data_['Date'].astype('datetime64')
    data_['Year'] = data_['Date'].dt.year
    data_['Month'] = data_['Date'].dt.month
    data_['SKU'] = sku
    nexgard_data = pd.concat([nexgard_data, data_])


# print(len(data))
# display(data.head(10))
# display(data.tail(10))
nexgard_data.loc[nexgard_data['Date']>='2021-09-01', 'Gross_Revenue'] = np.nan
nexgard_data['brand'] = 'Nexgard'

nexgard_data.to_csv("C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Nexgard\\Nexgard_sku_pred_agg.csv")



### SKU Prediction Aggregation data 
### Heartgard
heartgard_sku_lst = ["145448", "145449", "145450", "141348", "141354", "141360"]
print(heartgard_sku_lst)

sku = "145448"
heartgard_data = pd.read_csv(f"C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Heartgard\\{sku}.csv")
heartgard_data.rename(columns={sku: "Gross_Revenue"}, inplace=True)
heartgard_data['Date'] = heartgard_data['Date'].astype('datetime64')
heartgard_data['Year'] = heartgard_data['Date'].dt.year
heartgard_data['Month'] = heartgard_data['Date'].dt.month
heartgard_data['SKU'] = sku
# print(data.head(50))


for sku in heartgard_sku_lst[1:]:
    data_ = pd.read_csv(f"C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Heartgard\\{sku}.csv")
    data_.rename(columns={sku: "Gross_Revenue"}, inplace=True)
    data_['Date'] = data_['Date'].astype('datetime64')
    data_['Year'] = data_['Date'].dt.year
    data_['Month'] = data_['Date'].dt.month
    data_['SKU'] = sku
    heartgard_data = pd.concat([heartgard_data, data_])


# print(len(data))
# display(data.head(10))
# display(data.tail(10))
heartgard_data.loc[heartgard_data['Date']>='2021-09-01', 'Gross_Revenue'] = np.nan
heartgard_data['brand'] = 'Heartgard'

heartgard_data.to_csv("C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Heartgard\\Heartgard_sku_pred_agg.csv")

data = pd.concat([nexgard_data, heartgard_data])
display(data.tail())
data.to_csv("C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\brand_sku_pred_agg.csv")