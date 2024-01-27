import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Preperation

data = pd.read_csv("datasets/BankChurners.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = data.copy()
df.info()
df.isnull().sum()
df.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"],
    axis=1, inplace=True)
df.head()


# Defining Required Functions


def filter_values(dataframe, column, values):
    """
    Used to filter the desired data from the columns
    Parameters
    ----------
    dataframe : pandas dataframe
    column : Refers to the column in the selected dataframe
    values : Refers to the value to be filtered in the selected column

    Returns
    -------
    Returns the filtered result

    """
    filtered_dataframe = dataframe[dataframe[column] == values]
    return filtered_dataframe


def grouping(dataframe, columns, values, agg_process):
    """
    Used to basic group the selected values
    Parameters
    ----------
    dataframe : Pandas dataframe
    columns : Specifies the column to group
    values : Indicates which column will be processed in return for the column to be grouped
    agg_process : Refers to the action to be taken. For example: sum, count, mean

    Returns
    -------
    Returns the grouped table
    """
    grouped_dataframe = dataframe.groupby(columns).agg({values: agg_process})
    return grouped_dataframe


# Splitting the dataset into two parts, attrited and exsiting customers

attrited_customer = filter_values(df, "Attrition_Flag", "Attrited Customer")
existing_customer = filter_values(df, "Attrition_Flag", "Existing Customer")


# Analysis
def anaysing_relation(dataframe_a, dataframe_e, column_df_a, column_df_e):
    """
    The overall objective of the study is to find out why customers stop using the bank's credit card.
    The function therefore tries to find a structural difference between customers who use the credit card and customers
    who stop using it. It does this by creating visual graphs by proportioning each variable within each variable.
    Structural differences are then tried to be detected from the graph created.

    Parameters
    ----------
    dataframe_a : Customers who stopped using their credit card
    dataframe_e : Customers who continue using their credit card
    column_df_a : Queried column for attrited customers
    column_df_e : Queried column for existing customers

    Returns
    -------
    Ready to process tables for both groups

    """
    prepared_data_a = grouping(dataframe_a, column_df_a, column_df_a, "count")
    prepared_data_e = grouping(dataframe_e, column_df_e, column_df_e, "count")

    prepared_data_a.columns = ["Count"]
    prepared_data_a.reset_index(inplace=True)
    prepared_data_e.columns = ["Count"]
    prepared_data_e.reset_index(inplace=True)

    prepared_data_a["PER"] = [value / (sum(prepared_data_a["Count"])) for value in
                              prepared_data_a["Count"]]
    prepared_data_e["PER"] = [value / (sum(prepared_data_e["Count"])) for value in
                              prepared_data_e["Count"]]
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    sns.barplot(x=column_df_a,
                y="PER",
                ax=ax[0],
                palette="magma",
                data=prepared_data_a)
    ax[0].set_title(f"{column_df_a} Distribution For Existing Customers")
    sns.barplot(x=column_df_e,
                y="PER",
                ax=ax[1],
                palette="viridis",
                data=prepared_data_e)
    ax[1].set_title(f"{column_df_e} Distribution for Attrited Customers")
    plt.savefig(f"graphics/{column_df_a}_distribution.png")
    plt.show()
    return prepared_data_a, prepared_data_e


new_df = df.iloc[::, 2:]
new_df.drop(["Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Avg_Utilization_Ratio", "Total_Ct_Chng_Q4_Q1"],
            axis=1, inplace=True)

# Since some variables are very detailed, we cannot observe structural differences from the graphs.
# Therefore, a more detailed analysis is required


list_detail = ["Customer_Age", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct"]
# Categorize of Customers Age
cutoff_customers_age = [25, 45, 65, 85]
labels_customers_age = ["Young-Adult", "Adult", "Senior"]
new_df["ct_Customer_Age"] = pd.cut(new_df["Customer_Age"],
                                   bins=cutoff_customers_age,
                                   labels=labels_customers_age)

# Categorize of Total Amount Change
cutoff_total_amout_chng = [-1, 1.5, 2.5, 3.5]
labels_total_amout_chng = ["Low", "Medium", "Heigh"]

new_df["ct_Total_Amount_Chng"] = pd.cut(new_df["Total_Amt_Chng_Q4_Q1"],
                                        bins=cutoff_total_amout_chng,
                                        labels=labels_total_amout_chng)

# Categorize of Total Transaction Amout
cutoff_total_trans_amt = [0, 1000, 5000, 15000, 20000]
labels_total_trans_amt = ["Low", "Intermediate Low", "Intermediate", "Heigh"]

new_df["ct_Total_Trans_Amt"] = pd.cut(new_df["Total_Trans_Amt"],
                                      bins=cutoff_total_trans_amt,
                                      labels=labels_total_trans_amt)

# Categorize of Total Transaction Count

cutoff_total_trans_ct = [0, 50, 100, 150]
labes_total_trans_ct = ["Low", "Medium", "Heigh"]

new_df["ct_Total_Trans_Ct"] = pd.cut(new_df["Total_Trans_Ct"],
                                     bins=cutoff_total_trans_ct,
                                     labels=labes_total_trans_ct)

# Final analysing

new_df = new_df.drop(list_detail, axis=1)

for col in new_df.columns:
    anaysing_relation(attrited_customer, existing_customer, col, col)
