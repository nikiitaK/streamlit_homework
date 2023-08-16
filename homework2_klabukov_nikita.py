import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import streamlit as st


st.header("Step 1: upload your dataframe (only in 'utf-8')")
uploaded_file = st.file_uploader("Upload here...", type=['csv'])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


    st.header("Step 2: select two characteristics for analysis")
    names_and_types = [["--------", 0], ["--------", 0]]
    key_value = 0

    names_and_types[0][0] = st.selectbox('Select characteristics №1',
                            ["--------"]+list(dataframe.columns.values),
                            index=0, key=key_value)
    key_value += 1
    names_and_types[1][0] = st.selectbox('Select characteristics №2',
                            ["--------"]+list(dataframe.columns.values),
                            index=0, key=key_value)
    key_value += 1
    
    if names_and_types[0][0] != "--------" and names_and_types[1][0] != "--------":
        for col_i in range(2):

            column_name = names_and_types[col_i][0]
            if (len(pd.unique(dataframe[column_name])) <= 10) or (dataframe.dtypes[column_name] != 'float64' and dataframe.dtypes[column_name] != 'int64'):
                
                names_and_types[col_i][1] = "Categorial"
                st.header(column_name)
                st.write("Column type: ", dataframe.dtypes[column_name], 
                        ", unic data: ", len(pd.unique(dataframe[column_name])), 
                        ", most likely it is :green[_Categorial_] data")

                top_count = st.slider('Show top the most frequent values (top_10 - default, top_100 - max)', 1, 100, 10, key=key_value)
                top_values = dataframe[column_name].sort_values().value_counts(sort=False).nlargest(top_count)
                key_value += 1
                
                def pieChart():
                    fig = plt.figure(figsize=(10, 4))
                    plt.pie(top_values.values, labels = top_values.index)
                    st.pyplot(fig)

                pieChart()

            else:
                names_and_types[col_i][1] = "Numeric"
                st.header(column_name)
                st.write("Column type: ", dataframe.dtypes[column_name], 
                        ", unic data: ", len(pd.unique(dataframe[column_name])), 
                        ", most likely it is :green[_Numeric_] data")

                type_hist = st.radio("select the charts", 
                                    ('bar charts', 'area charts', 'line charts'), 
                                    index=1, key=key_value, horizontal=True)
                key_value += 1

                if type_hist == 'bar charts':
                    st.bar_chart(dataframe[column_name].sort_values().value_counts(sort=False))
                elif type_hist == 'area charts':
                    st.area_chart(dataframe[column_name].sort_values().value_counts(sort=False))
                else:
                    st.line_chart(dataframe[column_name].sort_values().value_counts(sort=False))



        st.header("Step 3: choose the test")

        dataframe.dropna(subset=[names_and_types[0][0], names_and_types[1][0]], inplace=True)
        alpha_value = st.number_input('alpha value', value=0.05)
        
        type_of_test = st.selectbox('Select test',
                                    ("--------", 
                                    "Mann-Whitney U-test (Numeric & Numeric)", 
                                    "Kolmogorov–Smirnov test (Numeric & Numeric)", 
                                    "Chi-square test (Categorial & Categorial)"),
                                    index=0)

        if type_of_test != "--------":
            if type_of_test == "Mann-Whitney U-test (Numeric & Numeric)" and names_and_types[0][1] == names_and_types[1][1] == "Numeric":
                test_result = stats.mannwhitneyu(dataframe[names_and_types[0][0]], dataframe[names_and_types[1][0]], alternative='two-sided')
                st.write(test_result)
                st.write("p-value:", test_result[1], "; alpha value:", alpha_value)
                st.write("H0: distribution underlying sample 'x' IS THE SAME as 'y' (p-value >= alpha) - ", test_result[1] >= alpha_value)
                st.write("HA: distribution underlying sample 'x' IS NOT THE SAME as 'y' (p-value < alpha) - ", test_result[1] < alpha_value)

            elif type_of_test == "Kolmogorov–Smirnov test (Numeric & Numeric)" and names_and_types[0][1] == names_and_types[1][1] == "Numeric":
                test_result = stats.ks_2samp(dataframe[names_and_types[0][0]], dataframe[names_and_types[1][0]])
                st.write(test_result)
                st.write("p-value:", test_result[1], "; alpha value:", alpha_value)
                st.write("H0: distribution underlying sample 'x' IS THE SAME as 'y' (p-value >= alpha) - ", test_result[1] >= alpha_value)
                st.write("HA: distribution underlying sample 'x' IS NOT THE SAME as 'y' (p-value < alpha) - ", test_result[1] < alpha_value)

            elif type_of_test == "Chi-square test (Categorial & Categorial)" and names_and_types[0][1] == names_and_types[1][1] == "Categorial":
                df0_name = dataframe[names_and_types[0][0]]
                df1_name = dataframe[names_and_types[1][0]]

                cross_tab = pd.crosstab(df0_name, df1_name, margins = True)
                cross_tab.columns = list((df1_name).unique())+["row_totals"]
                cross_tab.index = list(pd.unique(df0_name))+["col_totals"]
                st.write("Сross tab:", cross_tab)
                observed = cross_tab.iloc[0:len(pd.unique(df0_name)),0:len(pd.unique(df1_name))]

                expected =  np.outer(cross_tab["row_totals"][0:len(pd.unique(df0_name))],
                     cross_tab.loc["col_totals"][0:len(pd.unique(df1_name))]) / 1000
                expected = pd.DataFrame(expected)
                expected.columns = list(pd.unique(df1_name))
                expected.index = list(pd.unique(df0_name))
                st.write("Expected tab:", expected)

                degrees_of_freedom = (len(pd.unique(df0_name))-1)*(len(pd.unique(df1_name))-1)

                chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
                st.write("Chi squared stat:", chi_squared_stat)

                crit = stats.chi2.ppf(q = 1 - alpha_value,
                                     df = degrees_of_freedom)
                st.write("Degrees of freedom:", degrees_of_freedom)
                st.write("Critical value:", crit)

                p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,
                                             df=degrees_of_freedom)

                st.write("  ")

                ch_sq, pvalue, df, expected_ = stats.chi2_contingency(observed = observed)
                st.write("ch_sq, pvalue, df, expected_:")
                st.write(ch_sq, pvalue, df, expected_)

                st.write("  ")
                st.write("  ")
                st.write("  ")

                st.write("p-value:", pvalue, "; alpha value:", alpha_value)
                st.write("H0: the two variables are INDEPENDENT (p-value < alpha) - ", pvalue < alpha_value)
                st.write("HA: the two variables are NOT INDEPENDENT (p-value >= alpha) - ", pvalue >= alpha_value)

            else:
                st.error('This is an error, it looks like the method is not suitable for the used data types', icon="🚨")