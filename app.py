import streamlit as st
from snowflake.snowpark.context import get_active_session
import sys
import streamlit.components.v1 as components
import pickle as pkl
import pandas as pd
import os,sys
import lime
import lime.lime_tabular
from snowflake.snowpark.functions import *
from joblib import load
from pycaret.classification import setup, create_model, save_model, interpret_model, load_model, predict_model
from scipy import stats
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
import math
from fuzzywuzzy import process
import base64

col1,col2=st.columns(2)
col2.write(" ", font="Roboto 36")
col2.write(" ", font="Roboto 24")
col2.write(" **KIPITHON | AVIATOR 2.0** ", font="Roboto 24")
col1.image("https://www.matillion.com/cache/imager/logos/21082/kipi-logo_dc6961c2e8d5c5de4ce8389f1f2ad799.jpg",width=150)


def outlier_capping(df,x,first,second):
            percentile25 = df[x].quantile(first)
            percentile75 = df[x].quantile(second)
            iqr = percentile75 - percentile25

            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            df[x] = np.where(df[x] > upper_limit,upper_limit,df[x])
            df[x] = np.where(df[x] < lower_limit,lower_limit,df[x])
            return df

# Function to apply floor or ceil operation based on the values
def floor_ceil(value):
    if value - math.floor(value) < 0.5:
        return math.floor(value)
    else:
        return math.ceil(value)


def top_counts(df,x,num):
        # Lead Source Column For Top 9 & Others
        value_counts = df[x].value_counts()
        top_values = value_counts.head(num)
        others_count = value_counts.sum() - top_values.sum()
        top_values = top_values.reset_index()
        top_values.columns = [x, 'Count']
        others_df = pd.DataFrame({x: ['OTHERS'], 'Count': [others_count]})
        top_values = top_values.append(others_df, ignore_index=True)
        return top_values

def retain_top_n_categories(df, column_name, n,new_column_name):
    # Calculate frequency of each unique value in the column
    value_counts = df[column_name].value_counts()
    
    # Select top n most frequent values
    top_n_values = value_counts.head(n).index.tolist()
    
    # Create a new column to retain top n categories and flag others as "Others"
    df[new_column_name] = df[column_name].apply(lambda x: x if x in top_n_values else 'Others')
    return df
        
class EDA_Dataframe_Analysis():
    def __init__(self):
        print("General_EDA object created")

    def show_dtypes(self,x):
        return x.dtypes


    def show_columns(self,x):
        return x.columns


    def Show_Missing(self,x):
        return x.isna().sum()


    def show_hist(self,x):
        fig = plt.figure(figsize = (15,20))
        ax = fig.gca()
        return x.hist(ax=ax)


    def Tabulation(self,x):
        table = pd.DataFrame(x.dtypes,columns=['dtypes'])
        table1 =pd.DataFrame(x.columns,columns=['Names'])
        table = table.reset_index()
        table= table.rename(columns={'index':'Name'})
        table['No of Missing'] = x.isnull().sum().values    
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
        table['First Observation'] = x.loc[0].astype(str).values
        table['Second Observation'] = x.loc[1].astype(str).values
        table['Third Observation'] = x.loc[2].astype(str).values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
        return table


    def Numerical_variables(self,x):
        Num_var = [var for var in x.columns if x[var].dtypes!="object"and x[var].dtypes != "datetime64[ns]"]
        Num_var = x[Num_var]
        return Num_var

    def categorical_variables(self,x):
        cat_var = [var for var in x.columns if x[var].dtypes=="object"]
        cat_var = x[cat_var]
        return cat_var

        # def Show_pearsonr(self,x,y):
        #     result = pearsonr(x,y)
        #     return result

        
        # def Show_spearmanr(self,x,y):
        #     result = spearmanr(x,y)
        #     return result


    def plotly(self,a,x,y):
        fig = px.scatter(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                        line=dict(width=2,
                                                color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
        fig.show()

    def show_displot(self,x):
            plt.figure(1)
            plt.subplot(121)
            sns.distplot(x)


            plt.subplot(122)
            x.plot.box(figsize=(16,5))

            plt.show()

    def Show_DisPlot(self,x):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12,7))
        return sns.distplot(x, bins = 25)

    def Show_CountPlot(self,x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x,ax=ax)

    def plotly_histogram(self,a,x,y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                        line=dict(width=2,
                                                color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
        fig.show()


    def plotly_violin(self,a,x,y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                        line=dict(width=2,
                                                color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
        fig.show()

    def Show_PairPlot(self,x):
        return sns.pairplot(x)

    def Show_HeatMap(self,x):
        f,ax = plt.subplots(figsize=(15, 15))
        x = self.Numerical_variables(x)
        return sns.heatmap(x.corr(),annot=True,ax=ax);


    def label(self,x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x=le.fit_transform(x)
        return x

    def label1(self,x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x=le.fit_transform(x)
        return x

    def concat(self,x,y,z,axis):
        return pd.concat([x,y,z],axis)

    def dummy(self,x):
        return pd.get_dummies(x)


    def qqplot(self,x):
        return sm.qqplot(x, line ='45')


    def Anderson_test(self,a):
        return anderson(a)

    def PCA(self,x):
        pca =PCA(n_components=8)
        principlecomponents = pca.fit_transform(x)
        principledf = pd.DataFrame(data = principlecomponents)
        return principledf

    def outlier(self,x):
        high=0
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3-q1
        low = q1-1.5*iqr
        high += q3+1.5*iqr
        outlier = (x.loc[(x < low) | (x > high)])
        return(outlier)



    def check_cat_relation(self,x,y,confidence_interval):
        cross_table = pd.crosstab(x,y,margins=True)
        stat,p,dof,expected = chi2_contingency(cross_table)
        print("Chi_Square Value = {0}".format(stat))
        print("P-Value = {0}".format(p))
        alpha = 1 - confidence_interval
        return p,alpha
        if p > alpha:
            print(">> Accepting Null Hypothesis <<")
            print("There Is No Relationship Between Two Variables")
        else:
            print(">> Rejecting Null Hypothesis <<")
            print("There Is A Significance Relationship Between Two Variables")


class Attribute_Information():

    def __init__(self):
        
        print("Attribute Information object created")
        
    def Column_information(self,data):
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of observation'] = data.shape[0]
        data_info.loc[0,'No of Variables'] = data.shape[1]
        data_info.loc[0,'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0,'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0,'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0,'No of zero variance variables'] = data.loc[:,data.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info
    
    def __get_missing_values(self,data):
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

        
    def __iqr(self,x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self,x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self,df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
            i = i+1
        return data_info_num
    
    def statistical_summary(self,df):
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num



class Charts():

    def __init__(self):
        print("Charts object created")
    
    def scatter_plot(self,df,X,Y, Color=None):
        fig = px.scatter(df, y = Y, x=X,orientation='h', color=Color, render_mode='svg')
        st.plotly_chart(fig, use_container_width=True)
        # fig = sns.scatterplot(df, y = Y, x=X, hue=Color)
        # st.pyplot()

    def box_plot(self,df,X,Y):
        fig = px.box(df, y = Y, x=X)
        st.plotly_chart(fig, use_container_width=True)
        # fig = sns.lineplot(df, y = Y, x=X, hue=Color)
        # st.pyplot()

    def bar_plot(self,df, X, Color):
        fig = px.bar(df, x=X, color=Color)
        st.plotly_chart(fig, use_container_width=True)
        # fig = sns.lineplot(df, y = Y, x=X, hue=Color)
        # st.pyplot()

############################################MAIN CODE STARTS#######################################################################
def main():

    # #st.title("My Streamlit App with Logo")
    # # Retrieve logo image data from Snowflake
    # logo_data = get_logo_from_snowflake()
    # # Convert logo image data to base64
    # base64_logo = convert_to_base64(logo_data)
    # # Display logo in Streamlit
    # st.image(base64_logo, caption='My Logo')

    
    st.title("Flight Departure Delay Prediction")
    tabs = st.tabs(['HOME','EDA', 'Features','Model Performance','Bulk Inferencing','Single Inferencing'])
    st.info(" Streamlit Web Application ") 


    with tabs[0]:
        st.info("""**Problem Statement**""")

        prob_stat = """+ Flight delays  are a major source of frustration for an airline company, resulting in significant losses for airlines in terms of customer satisfaction and loyalty."""
        st.markdown(prob_stat)

        st.markdown("""+ One of the main factor affecting flight delay is weather. Understanding the affect of weather conditions during future flight take off schedules and taking precautionay measures are critical in ensuring smooth operations and high customer satisfaction.""")

        st.info("""**Impact of solving the problem to the Business**""")

        st.markdown("""+ Improved Operational Efficiency: By predicting flight delays, airlines can proactively adjust their operations to minimize the impact of delays. This includes rerouting flights, re-assigning crew and aircraft, and adjusting ground staff schedules.""")

        st.markdown("""+ Enhanced Customer Experience: Airlines can provide passengers with more accurate and timely information about flight delays, allowing them to plan accordingly. This can lead to increased customer satisfaction and loyalty.""")

        
        st.markdown("""+ Reduced Costs: Flight delays can result in significant costs, including fuel, crew overtime, and passenger compensation. By predicting delays, airlines can take steps to mitigate these costs.""")
        
        st.markdown("""+ Improved Safety: By predicting flight delays, airlines can identify potential safety risks and take steps to mitigate them. For example, they can adjust flight plans to avoid adverse weather conditions. """)

        
        c1,c2,c3=st.columns([0.1,0.8,0.1])
        c2.image("https://projects.smartinternz.com/cdn/shop/files/image.jpg")

        st.info("""**Impact of Solving the Business Problem**""")

        st.markdown("""+ Enhanced Collaboration: Flight delay prediction can facilitate collaboration between airlines, airports, and air traffic control authorities. This can lead to more efficient management of air traffic and reduced delays. """)
        st.markdown("""+ Competitive Advantage: Airlines that can accurately predict flight delays can gain a competitive advantage over their rivals. This can lead to increased market share and revenue. """)
        
        #c5,c6=st.columns([0.95,0.025])
    
        #c5.image("https://www.google.com/search?sca_esv=99636c5107a06ae2&rlz=1C1VDKB_enIN1097IN1097&q=customer+churn+prediction&uds=AMwkrPvrOAvos78kDpVSUxKLk03VeVQdJXJwyW0Zegd7VTAv_MVCz_xxWwGekgPTRe48FB0ux9WK1q6JiZAdW4DrHIA1FhqPQ9-ZMqVJ064yUlw4JmFb84qUX-9__sgOHqpMdd6Uj-aJ63IRmyWtTXQTjYpcX72Q2aAd0T9D1_VSvpPifMupfGioQhXNeMldHt2JZXDSDFcaXtSxg8xbszi8UkJcHmxDk1oVOiZIfEgvpIT2gFqTvm9Xzi5qwmu-IUv1Ze-ZNd4ZzwwQEek3nT-cpp-d-9ICfYUbWPAxfuxn4wiI6QBtKszPNCEWuQfRmNwb-626Qvej&udm=2&prmd=invsbmtz&sa=X&sqi=2&ved=2ahUKEwj77K72vMaFAxVQR2wGHenZDnEQtKgLegQIERAB&biw=1422&bih=701&dpr=1.5#vhid=QxVUZwClWUVPkM&vssid=mosaic")
    with tabs[1]:

            st.subheader("Exploratory Data Analysis")
            df= session.table('RAW_DELAY_SCHEMA.DELAY_DATA').to_pandas()
            df = df.drop(['TIME'],axis=1)
        
            
            if st.checkbox("Show dtypes"):
                    # Customize column names
                    #df_custom = df.rename(columns={'Column1': 'COLUMN NAME', 'Column2': 'DATA TYPE'})

        # Customize table size
                    #table_width = st.slider("Table Width", 100, 1000, 500)

        # Display DataFrame with customized column names and size
                    #st.write("Customized DataFrame:")
                    #st.dataframe(df_custom, width=table_width)
                    st.write("Data Types of Columns:")
                    st.write(dataframe.show_dtypes(df))

            if st.checkbox("Show Columns"):
                st.write(dataframe.show_columns(df))

            if st.checkbox("Show Missing"):
                st.write(dataframe.Show_Missing(df))

            if st.checkbox("Column information"):
                st.write(info.Column_information(df))

            if st.checkbox("Num Count Summary"):
                #df_c1 = df.drop('ISFIRSTTIMEBUYER',axis=1)
                st.write(info.num_count_summary(df))

            if st.checkbox("Statistical Summary"):
                #df_c2 = df.drop('ISFIRSTTIMEBUYER',axis=1)
                st.write(info.statistical_summary(df))      


                
            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns",dataframe.show_columns(df))
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Numerical Variables"):
                num_df = dataframe.Numerical_variables(df)
                numer_df=pd.DataFrame(num_df)                
                st.dataframe(numer_df)

            if st.checkbox("Categorical Variables"):
                new_df = dataframe.categorical_variables(df)
                catego_df=pd.DataFrame(new_df)                
                st.dataframe(catego_df)

            # Set Seaborn theme
            sns.set_style("darkgrid")
            sns.set_palette("bright")
    
             #-----------------------------------univariate-------------------------

            # df = outlier_capping(df,'BALANCE_AT_CLOSING',0.05,0.95)
            # df = outlier_capping(df,'CURRENT_BALANCE',0.25,0.75)
            # df = outlier_capping(df,'AGE',0.25,0.75)
            # df = outlier_capping(df,'ACCOUNT_AGE',0.25,0.75)
            # # df = outlier_capping(df,'OD_FREQUENCY',0.05,0.95)
            # df = outlier_capping(df,'TOTAL_TRAN_COUNT',0.25,0.75)
            # df = outlier_capping(df,'AVERAGE_TRAN_CREDIT_AMT',0.25,0.75)
            # df = outlier_capping(df,'CNT_CREDIT_TRAN',0.25,0.75)
            # df = outlier_capping(df,'AVERAGE_TRAN_DEBIT_AMT',0.25,0.75)
            # df = outlier_capping(df,'CNT_DEBIT_TRAN',0.25,0.75)
            # df = outlier_capping(df,'DAYS_SINCE_MOST_RECENT_TRAN',0.25,0.75)

            # df['OD_FREQUENCY_PER_YEAR'] = df['OD_FREQUENCY_PER_YEAR'].apply(floor_ceil)
        
            def pie_plot(df, names):
                fig = px.pie(df, names=names, hole = 0.3)
                fig.update_layout(title={'text':f"{names} Distribution", 'x': 0.5, 'y':0.95})
                st.plotly_chart(fig, use_container_width=True)

            def bar_plot(self,df, X, Color):
                fig = px.bar(df, x=X, color=Color)
                fig.update_layout(title={'text':f"{X} vs {Color}", 'x': 0.5, 'y':0.95}, margin= dict(l=0,r=10,b=10,t=30), yaxis_title=Color, xaxis_title=X)
                st.plotly_chart(fig, use_container_width=True)

            def bar_plot_XY(df, X, Y, Color):
                fig = px.bar(df, x=X, y=Y, color=Color)
            
                # Calculate the percentage values
                total_counts = df.groupby(X)[Y].sum().reset_index()
                total_counts['percentage'] = (total_counts[Y] / total_counts[Y].sum()) * 100
            
                # Add percentage labels to the bars
                for i, row in total_counts.iterrows():
                    percentage_label = f"{row['percentage']:.2f}%"
                    fig.add_annotation(
                        x=row[X],
                        y=row[Y],
                        text=percentage_label,
                        showarrow=False,
                        font=dict(size=10),
                        yshift=10
                    )
            
                fig.update_layout(
                    title={'text': f"{X} by {Y}", 'x': 0.5, 'y': 0.95},
                    margin=dict(l=0, r=10, b=10, t=30),
                    yaxis_title=Color,
                    xaxis_title=X,
                )
            
                st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Univariate Analysis"):

                df = df[df.CHURN==1]
                x = st.selectbox("Select Category to view distribution", [ 'HOUR', 
                                'FORECAST', 'TEMPERATURE', 'CLOUD_COVER', 'CLOUD_CEILING', 'VISIBILITY', 
                                'PRECIPITATION', 'WIND', 'WIND_GUSTS'
                                    ])
                
                if x=='CLOUD_CEILING':
                    top_df = top_counts(df,'AIRLINE',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)
                
                if x=='WIND':
                    top_df = top_counts(df,'AIRLINE',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='WIND_GUSTS':
                    top_df = top_counts(df,'AIRLINE',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='VISIBILITY':
                    top_df = top_counts(df,'SOURCE_CITY',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='PRECIPITATION':
                    top_df = top_counts(df,'DATE',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='HOUR':
                    top_df = top_counts(df,'HOUR',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='FORECAST':
                    top_df = top_counts(df,'FORECAST',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='TEMPERATURE':
                    top_df = top_counts(df,'TEMPERATURE',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)

                if x=='CLOUD_COVER':
                    top_df = top_counts(df,'CLOUD_COVER',9)
                    fig = px.pie(top_df, values='Count', names=x, hole = 0.3)
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    st.plotly_chart(fig, use_container_width=True)
                    
                # if x =='TYPE':
                #     pie_plot(df=df, names = x)
                        

                if x in ('TENURE','WAREHOUSETOHOME','HOURSPENDONAPP', 'NUMBEROFDEVICEREGISTERED',
                        'SATISFACTIONSCORE','NUMBEROFADDRESS','ORDERAMOUNTHIKEFROMLASTYEAR','COUPONUSED',
                                    'ORDERCOUNT','DAYSINCELASTORDER','CASHBACKAMOUNT'):
                    # fig, ax = plt.subplots()
                    # bars=sns.histplot(data=df, x=x, bins=8)
                    fig = px.histogram(df, x=x, nbins=10,histnorm ='percent',text_auto=True)
                    #fig.update_xaxes(range=[0, df[x].max()])
                    fig.update_layout(
                        yaxis_title_text='PERCENTAGE', # yaxis label
                    )
                    fig.update_layout(title={'text':f"{x} Distribution", 'x': 0.5, 'y':0.95})
                    fig.update_traces(texttemplate='%{y:.0f}', textposition='outside')

                    st.plotly_chart(fig)

                if x== 'TENURE':
                	st.write("71% of the customers are using this application for tenure between 0 to 5 years.")
                if x== 'PREFERREDLOGINDEVICE':
                	st.write("65.8% of the customers are prefered login using mobile phones.")
                if x== 'CITYTIER':
                	st.write("56.1% of the customers living in tier 1 city.")
                if x== 'WAREHOUSETOHOME':
                	st.write("27% of the customers have distance between warehouse to home is 15 to 20 KM.")
                if x== 'PREFERREDPAYMENTMODE':
                	st.write("37.6% of the customers prefered payment mode is debit card.")
                if x== 'GENDER':
                	st.write("63.3% of the customers are Male.")
                if x== 'HOURSPENDONAPP':
                	st.write("55% of the customers spend 3 hours on application per day")
                if x== 'NUMBEROFDEVICEREGISTERED':
                	st.write("41% of the customers registered in 4 devices.")
                if x== 'PREFEREDORDERCAT':
                	st.write("60.1% of the customers prefered to order in mobile phone category")
                if x== 'SATISFACTIONSCORE':
                	st.write("28% of the customers are given 5 star ratings.")
                if x== 'MARITALSTATUS':
                	st.write("50.6% of the customers are single")
                if x== 'NUMBEROFADDRESS':
                	st.write("49% of the customers have added 2 to 4 addresses in application")
                if x== 'COMPLAIN':
                	st.write("53.6% of the customers have registered thier complaints")
                if x== 'ORDERAMOUNTHIKEFROMLASTYEAR':
                	st.write("29% of the customer have increased thier orders from 12% to 14% in 1 year.")
                if x== 'COUPONUSED':
                	st.write("59% of the customers are used 1 to 2 coupons to order")
                if x== 'ORDERCOUNT':
                	st.write("48% of the customers placed 2 to 4 orders in 1 month")
                if x== 'DAYSINCELASTORDER':
                	st.write("70% of the customers have not orderd in last 5 days")
                if x== 'CASHBACKAMOUNT':
                	st.write("50% of the customers received cashback of Rs 100 to Rs 150.")

                # if x== 'BALANCE_AT_CLOSING':
                #     st.write("68% of the accounts had balance at closing less than 250 USD.")
                # if x== 'TYPE':
                #     st.write("77.9% of the accounts churned had type as Checking accounts.")
                # if x== 'ACCOUNT_DESCRIPTION':
                #     st.write("21.4% of the accounts had an account description as Basic Business Checking.")
                # if x== 'AGE':
                #     st.write("29% customers are in the age bracket between 40-60.")
                # if x== 'ACCOUNT_AGE':
                #     st.write("32% of the accounts were opened less than 5 years back.")
                # if x== 'OD_FREQUENCY_PER_YEAR':
                #     st.write("98% of the accounts had an overdraft frequency (total overdraft days per account age) per year less than 20.")
                # if x== 'TOTAL_TRAN_COUNT':
                #     st.write("58% of the accounts had a total transaction count less than 250.")
                # if x== 'MOST_OCCURING_TRAN_DESCRIPTION':
                #     st.write("35.5% of the accounts had most appearing transaction description as POS Description.")
                # if x== 'AVERAGE_TRAN_CREDIT_AMT':
                #     st.write("48% of the accounts had an average value of credit transactions lesser than 100 USD.")
                # if x== 'CNT_CREDIT_TRAN':
                #     st.write("72% of the accounts had total credit transactions lesser than 100.")
                # if x== 'AVERAGE_TRAN_DEBIT_AMT':
                #     st.write("44% of the accounts had an average value of debit transactions lesser than 100 USD.")
                # if x== 'CNT_DEBIT_TRAN':
                #     st.write("71% of the accounts had total debit transactions lesser than 500.")
                # if x== 'CREDIT_TO_DEBIT_RATIO':
                #     st.write("99% of the accounts had their credit-to-debit ratio lesser than 2.5.")
                # if x== 'DAYS_SINCE_MOST_RECENT_TRAN':
                #     st.write("71% of the churned accounts didn’t perform any transaction for more than 1 year.")
                



            if st.checkbox("Bivariate Analysis by CHURN"):
                # fig = px.histogram(df_g, x=x_column, color="CLOSED_CAT",histnorm ='percent',text_auto=True)
                # st.plotly_chart(fig)
                x_column = st.selectbox("Select Category Default ", ['TENURE', 'PREFERREDLOGINDEVICE', 'CITYTIER', 'WAREHOUSETOHOME', 
                                'PREFERREDPAYMENTMODE', 'GENDER', 'HOURSPENDONAPP', 'NUMBEROFDEVICEREGISTERED', 'PREFEREDORDERCAT', 
                                'SATISFACTIONSCORE', 'MARITALSTATUS', 'NUMBEROFADDRESS', 'COMPLAIN', 'ORDERAMOUNTHIKEFROMLASTYEAR', 'COUPONUSED', 
                                'ORDERCOUNT', 'DAYSINCELASTORDER', 'CASHBACKAMOUNT'
                                    ])
    
                df_top = retain_top_n_categories(df, 'PREFERREDLOGINDEVICE', 9,'PREFERREDLOGINDEVICE')
                #df_top = retain_top_n_categories(df_top, 'MOST_OCCURING_TRAN_DESCRIPTION', 9,'MOST_OCCURING_TRAN_DESCRIPTION')
                df_top['CHURN_CAT'] = df_top['CHURN'].map({0: 'NOT CHURN', 1: 'CHURN'})
                df_g = df_top.groupby([x_column, 'CHURN_CAT']).size().reset_index()
                df_g['CHURN_CAT'] = df_g['CHURN_CAT'].astype(str)
                df_g['percentage'] = df.groupby([x_column, 'CHURN_CAT']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
                df_g.columns = [x_column, 'CHURN_CAT', 'Count', 'Percentage']


       #          CATEGORICAL_COLS = ['PREFERREDLOGINDEVICE', 'PREFERREDPAYMENTMODE', 'GENDER', 'PREFEREDORDERCAT', 'MARITALSTATUS']

       #          NUMERICAL_COLS = ['CHURN', 'TENURE',  'CITYTIER',
       # 'WAREHOUSETOHOME',  'HOURSPENDONAPP',
       # 'NUMBEROFDEVICEREGISTERED',  'SATISFACTIONSCORE',
       #  'NUMBEROFADDRESS', 'COMPLAIN',
       # 'ORDERAMOUNTHIKEFROMLASTYEAR', 'COUPONUSED', 'ORDERCOUNT',
       # 'DAYSINCELASTORDER', 'CASHBACKAMOUNT']

                
                if x_column in ('PREFERREDLOGINDEVICE', 'PREFERREDPAYMENTMODE', 'GENDER', 'PREFEREDORDERCAT', 'MARITALSTATUS'):
                    fig = px.bar(df_g, x=x_column, y="Count",color='CHURN_CAT', text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
                    st.plotly_chart(fig)

                if x_column in ('TENURE',  'CITYTIER','WAREHOUSETOHOME',  'HOURSPENDONAPP','NUMBEROFDEVICEREGISTERED',  'SATISFACTIONSCORE',
                        'NUMBEROFADDRESS', 'COMPLAIN','ORDERAMOUNTHIKEFROMLASTYEAR', 'COUPONUSED', 'ORDERCOUNT',
                            'DAYSINCELASTORDER', 'CASHBACKAMOUNT'):
                    fig, ax = plt.subplots()
                    
                    bars=sns.histplot(data=df_top, x=x_column,hue="CHURN_CAT", bins=10,multiple='stack')
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    # Format x-axis labels to display values in millions
                    # def format_millions(x, pos):
                    #     return f'{x / 1_000_000:.1f}M'

                    # formatter = FuncFormatter(format_millions)
                    # bars.xaxis.set_major_formatter(formatter)
                    # sns.move_legend(ax, "right")
                    bars_height=0
                    for bar in bars.patches:
                        bars_height+=bar.get_height()
                        
                    # Add percentage values for each bar at the top
                    for bar in bars.patches:
                        yval = bar.get_height()
                        percentage_value = yval / bars_height * 100
                        if percentage_value > 0:
                            percentage = f'{yval / bars_height * 100:.1f}%'
                            ax.text(bar.get_x() + bar.get_width() / 2, yval+(yval*0.35), percentage, ha='center', va='bottom', fontsize=10)

                    st.pyplot(fig)
                # if x_column =='DAYS_SINCE_MOST_RECENT_TRAN':
                #     fig, ax = plt.subplots()
                    
                    # bars=sns.histplot(data=df_top, x=x_column,hue="CHURN_CAT", bins=10,multiple='stack')
                    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    # # Format x-axis labels to display values in millions
                    # # def format_millions(x, pos):
                    # #     return f'{x / 1_000_000:.1f}M'

                    # # formatter = FuncFormatter(format_millions)
                    # # bars.xaxis.set_major_formatter(formatter)
                    # # sns.move_legend(ax, "right")
                    

                    # bars_height=0
                    # for bar in bars.patches:
                    #     bars_height+=bar.get_height()
                        
                    # # Add percentage values for each bar at the top
                    # for bar in bars.patches:
                    #     yval = bar.get_height()
                    #     percentage_value = yval / bars_height * 100
                    #     if percentage_value > 0.8:
                    #         percentage = f'{yval / bars_height * 100:.1f}%'
                    #         ax.text(bar.get_x() + bar.get_width() / 2, yval+(yval*0.02), percentage, ha='center', va='bottom', fontsize=10)

                    # st.pyplot(fig)

                # if x_column =='TOTAL_ASSETS':
                #         fig, ax = plt.subplots()

                #         bars=sns.histplot(data=df_top, x=x_column,hue="CLOSED_CAT", bins=[0,20000,40000,60000,80000,100000,120000,140000],multiple='stack')
                #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                #         # Format x-axis labels to display values in millions
                #         def format_millions(x, pos):
                #             return f'{x / 1_000:.0f}K'
    
                #         formatter = FuncFormatter(format_millions)
                #         bars.xaxis.set_major_formatter(formatter)
                #         sns.move_legend(ax, "right")
                #         bars_height=0
                #         for bar in bars.patches:
                #             bars_height+=bar.get_height()
                        
                #         # Add percentage values for each bar at the top
                #         for bar in bars.patches:
                #             yval = bar.get_height()
                #             percentage = f'{yval / bars_height * 100:.1f}%'
                #             ax.text(bar.get_x() + bar.get_width() / 2, yval+(yval*0.08), percentage, ha='center', va='bottom', fontsize=10)
                        
                #         st.pyplot(fig)

     


            conclusion_paragraph = """
               **Conclusion**
                
We observe that the dataset encompasses **Churn features related to Customer and their respective transactions.**
It includes variables related to the Customer and transactions, such as the Customer - City tier,Preferred payment method,Gender,Preferred order category,Day Since last order  etc.
                
The **churned accounts** have shown distinctive characteristics like:
- 71% of the customers are using this application for tenure between 0 to 5 years.
- 70% of the customers have not orderd in last 5 days
- 65.8% of the customers are prefered login using mobile phones.
- 63.3% of the customers are Male.
- 60.1% of the customers prefered to order in mobile phone category
- 59% of the customers are used 1 to 2 coupons to order
- 56.1% of the customers living in tier 1 city.
- 55% of the customers spend 3 hours on application per day
- 53.6% of the customers have registered thier complaints
- 50.6% of the customers are single
- 50% of the customers received cashback of Rs 100 to Rs 150.
- 49% of the customers have added 2 to 4 addresses in application
- 48% of the customers placed 2 to 4 orders in 1 month
- 41% of the customers registered in 4 devices.
- 37.6% of the customers prefered payment mode is debit card.
- 29% of the customer have increased thier orders from 12% to 14% in 1 year.
- 28% of the customers are given 5 star ratings.
- 27% of the customers have distance between warehouse to home is 15 to 20 KM.
	
	

"""

            st.markdown(conclusion_paragraph)

    with tabs[2]:
        st.subheader('Global Explanability')
        session.file.get('@"CUST_CHURN"."CUSTOMER"."MODEL_STAGE"/Feature Importance.png','/tmp')
        st.image('/tmp/Feature Importance.png')
        st.write("If you look at all the feature on Y-axis , COMPLAIN is the most impactful parameter in calculating the probability of an customer getting churned. ")

    
    with tabs[3]:
        st.markdown('**Training Data Metrics**')
        model_results = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_MODEL_RESULTS').to_pandas()
        st.write(model_results)

        st.markdown('**Testing Data Metrics**')
        test_results = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_MODEL_RESULTS_TEST').to_pandas()
        st.write(test_results)

    with tabs[4]:
        db = st.selectbox("Select Database", ['CUST_CHURN','SNOWPARK', 'SNOWFLAKE', 'SNOWFLAKE_SAMPLE_DATA'])
        schema = st.selectbox("Select Schema", ['CUSTOMER','PUBLIC', 'INFORMATION_SCHEMA', 'PUBLIC'])
        table = st.selectbox("Select Table", ['CHURN_TEST_DATA','BULK_TABLE','SAMPLE_DATA'])

        session.file.get('@"CUST_CHURN"."CUSTOMER"."MODEL_STAGE"/churn_model.pkl','/tmp')
        model = load_model('/tmp/churn_model')

        
        if db == 'CUST_CHURN' and schema == 'CUSTOMER' and table == 'CHURN_TEST_DATA':
            tablename = f"{db}.{schema}.{table}"
            table_df = session.table(tablename)
            table_pd = table_df.to_pandas()
            submit = st.button("Predict for all")
            if submit:
                # prediction = session.sql(f"select FIDELITY_PREDICTION(BALANCE_AT_CLOSING, TYPE, ACCOUNT_DESCRIPTION, AGE, ACCOUNT_AGE, OD_FREQUENCY_PER_YEAR, TOTAL_TRAN_COUNT, MOST_OCCURING_TRAN_DESCRIPTION, AVERAGE_TRAN_CREDIT_AMT, CNT_CREDIT_TRAN, AVERAGE_TRAN_DEBIT_AMT, CNT_DEBIT_TRAN, CREDIT_TO_DEBIT_RATIO, DAYS_SINCE_MOST_RECENT_TRAN) as PRED from {tablename}")
                # labels = prediction.to_pandas()
                # #table_pd.insert(loc=1, column='CLUSTER_NUMBER', value= clusters['CLUST'])
                # table_pd["PREDICTED_CLOSED"] = labels['PRED']
                # st.dataframe(table_pd)

                predictions = predict_model(model, data=table_pd ,raw_score=True)
                predictions = predictions.drop(['prediction_label','prediction_score_0'],axis=1)
                predictions = predictions.rename(columns={'prediction_score_1':'PREDICTED_CLOSED'})
                predictions['PREDICTED_CLOSED'] = (predictions['PREDICTED_CLOSED']*100).round(2)
                st.dataframe(predictions)
                
        if st.checkbox("Local Explanability"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            table_pd = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_TEST_DATA').to_pandas()
            choice = st.selectbox("Select CUSTOMER ID  to view local explanability", [51918,55535,
                           51568,53566,50793,51134,53795,54200,51079])

            #one = table_pd.iloc[1:2].drop(['CUSTOMERID','CHURN'],axis=1)
            one = table_pd[table_pd.CUSTOMERID==choice].drop(['CUSTOMERID','CHURN'],axis=1)
            #st.write(one)
            instance_to_predict = model[:-1].transform(one)
    
            df = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_DATA_CLEAN').to_pandas()
            X = model[:-1].transform(df).drop(['CUSTOMERID','CHURN'], axis=1)
            X_featurenames = X.columns.tolist()
            explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                 feature_names=X_featurenames, 
                                 class_names=['NOT CHURNED','CHURNED'],
                                 discretize_continuous = True,
                                 verbose=True, mode='classification')
            
            exp = explainer.explain_instance(instance_to_predict.values.flatten(), model.named_steps["trained_model"].predict_proba,labels=(1,))
    
            st.subheader('Local Explanability of Prediction')
            exp.as_pyplot_figure()
            st.pyplot()
        # if st.checkbox("Local Explanability"):
        #     table_pd = session.sql('select * from CUST_CHURN.CUSTOMER.BULK_TABLE').to_pandas()
        #     choice = st.selectbox("Select Account Number to view local explanability", ['2114fd7143df749916de6551f52f2ec8','b4d9a4929bc52c21a3d866914b20cbee',
        #                    'e416b182768260600103c457d7bb8af0','accaec24edd4e430a10d36bd57ecedb6',
        #                    '7be1507eab2d491a7adb4ce0fbd3dc43','79001b2ff3177636e76823a36ae776dd',
        #                    '4b8cc22b5a50964666b0d6f81cbd11b5','9fb0ac07ff0682112b4059e16ac19f6f'])

        #     one = table_pd[table_pd.CUSTOMERID==choice].drop(['CUSTOMERID','CHURN'],axis=1)
        #     instance_to_predict = model[:-1].transform(one)
    
        #     df = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_DATA_CLEAN').to_pandas()
        #     X = model[:-1].transform(df).drop(['CUSTOMERID','CHURN'], axis=1)
        #     X_featurenames = X.columns.tolist()
        #     explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
        #                          feature_names=X_featurenames, 
        #                          class_names=['NOT CHURNED','CHURNED'],
        #                          discretize_continuous = True,
        #                          verbose=True, mode='classification')
            
        #     exp = explainer.explain_instance(instance_to_predict.values.flatten(), model.named_steps["trained_model"].predict_proba,labels=(1,))
    
        #     st.subheader('Local Explanability of Prediction')
        #     exp.as_pyplot_figure()
        #     st.pyplot()
                    
                
        
    with tabs[5]:
        df = session.sql('select * from CUST_CHURN.CUSTOMER.CHURN_DATA_CLEAN').to_pandas()
        columns = [ "TENURE", "PREFERREDLOGINDEVICE", "CITYTIER", "WAREHOUSETOHOME", "PREFERREDPAYMENTMODE",
                   "GENDER", "HOURSPENDONAPP", "NUMBEROFDEVICEREGISTERED", "PREFEREDORDERCAT", "SATISFACTIONSCORE", "MARITALSTATUS",
                   "NUMBEROFADDRESS", "COMPLAIN", "ORDERAMOUNTHIKEFROMLASTYEAR", "COUPONUSED", "ORDERCOUNT", "DAYSINCELASTORDER",
                   "CASHBACKAMOUNT", "CHURN"]

        df = df[columns]

        # BALANCE_AT_CLOSING_options =df['BALANCE_AT_CLOSING'].unique().tolist()
        # TYPE_options =df['TYPE'].unique().tolist()
        # ACCOUNT_DESCRIPTION_options =df['ACCOUNT_DESCRIPTION'].unique().tolist()
        # AGE_options =df['AGE'].unique().tolist()
        # ACCOUNT_AGE_options =df['ACCOUNT_AGE'].unique().tolist()
        # OD_FREQUENCY_PER_YEAR_options =df['OD_FREQUENCY_PER_YEAR'].unique().tolist()
        # #RETURNED_CHARGES_RATE_options =df['RETURNED_CHARGES_RATE'].unique().tolist()
        # TOTAL_TRAN_COUNT_options =df['TOTAL_TRAN_COUNT'].unique().tolist()
        # MOST_OCCURING_TRAN_DESCRIPTION_options =df['MOST_OCCURING_TRAN_DESCRIPTION'].unique().tolist()
        # AVERAGE_TRAN_CREDIT_AMT_options =df['AVERAGE_TRAN_CREDIT_AMT'].unique().tolist()
        # CNT_CREDIT_TRAN_options =df['CNT_CREDIT_TRAN'].unique().tolist()
        # AVERAGE_TRAN_DEBIT_AMT_options =df['AVERAGE_TRAN_DEBIT_AMT'].unique().tolist()
        # CNT_DEBIT_TRAN_options =df['CNT_DEBIT_TRAN'].unique().tolist()
        # CREDIT_TO_DEBIT_RATIO_options =df['CREDIT_TO_DEBIT_RATIO'].dropna().unique().tolist()
        # DAYS_SINCE_MOST_RECENT_TRAN_options =df['DAYS_SINCE_MOST_RECENT_TRAN'].unique().tolist()

        TENURE_options =df['TENURE'].unique().tolist()
        PREFERREDLOGINDEVICE_options =df['PREFERREDLOGINDEVICE'].unique().tolist()
        CITYTIER_options =df['CITYTIER'].unique().tolist()
        WAREHOUSETOHOME_options =df['WAREHOUSETOHOME'].unique().tolist()
        PREFERREDPAYMENTMODE_options =df['PREFERREDPAYMENTMODE'].unique().tolist()
        GENDER_options =df['GENDER'].unique().tolist()
        HOURSPENDONAPP_options =df['HOURSPENDONAPP'].unique().tolist()
        NUMBEROFDEVICEREGISTERED_options =df['NUMBEROFDEVICEREGISTERED'].unique().tolist()
        PREFEREDORDERCAT_options =df['PREFEREDORDERCAT'].unique().tolist()
        SATISFACTIONSCORE_options =df['SATISFACTIONSCORE'].unique().tolist()
        MARITALSTATUS_options =df['MARITALSTATUS'].unique().tolist()
        NUMBEROFADDRESS_options =df['NUMBEROFADDRESS'].unique().tolist()
        COMPLAIN_options =df['COMPLAIN'].dropna().unique().tolist()
        ORDERAMOUNTHIKEFROMLASTYEAR_options =df['ORDERAMOUNTHIKEFROMLASTYEAR'].unique().tolist()
        COUPONUSED_options =df['COUPONUSED'].unique().tolist()
        ORDERCOUNT_options =df['ORDERCOUNT'].unique().tolist()
        DAYSINCELASTORDER_options =df['DAYSINCELASTORDER'].dropna().unique().tolist()
        CASHBACKAMOUNT_options =df['CASHBACKAMOUNT'].unique().tolist()

        target_col = 'CHURN'

        st.title('Data Entry Form')
        # # Dropdowns for categorical values
        # BALANCE_AT_CLOSING = st.slider('Select BALANCE_AT_CLOSING', min_value=int(sorted(BALANCE_AT_CLOSING_options)[0]), max_value=int(sorted(BALANCE_AT_CLOSING_options, reverse=True)[0]))
        # TYPE = st.selectbox('Select TYPE', TYPE_options)
        # ACCOUNT_DESCRIPTION = st.selectbox('Select ACCOUNT_DESCRIPTION', ACCOUNT_DESCRIPTION_options)
        # AGE = st.slider('Select AGE', min_value=int(sorted(AGE_options)[0]), max_value=int(sorted(AGE_options, reverse=True)[0]))
        # ACCOUNT_AGE = st.slider('Select ACCOUNT_AGE', min_value=int(sorted(ACCOUNT_AGE_options)[0]), max_value=int(sorted(ACCOUNT_AGE_options, reverse=True)[0]))
        # OD_FREQUENCY_PER_YEAR = st.slider('Select OD_FREQUENCY_PER_YEAR_options', min_value=int(sorted(OD_FREQUENCY_PER_YEAR_options)[0]), max_value=int(sorted(OD_FREQUENCY_PER_YEAR_options, reverse=True)[0]))
        # #RETURNED_CHARGES_RATE = st.slider('Select RETURNED_CHARGES_RATE', min_value=int(sorted(RETURNED_CHARGES_RATE_options)[0]), max_value=int(sorted(RETURNED_CHARGES_RATE_options, reverse=True)[0]))
        # TOTAL_TRAN_COUNT = st.slider('Select TOTAL_TRAN_COUNT', min_value=int(sorted(TOTAL_TRAN_COUNT_options)[0]), max_value=int(sorted(TOTAL_TRAN_COUNT_options, reverse=True)[0]))
        # MOST_OCCURING_TRAN_DESCRIPTION = st.selectbox('Select MOST_OCCURING_TRAN_DESCRIPTION', MOST_OCCURING_TRAN_DESCRIPTION_options)
        # AVERAGE_TRAN_CREDIT_AMT = st.slider('Select AVERAGE_TRAN_CREDIT_AMT', min_value=int(sorted(AVERAGE_TRAN_CREDIT_AMT_options)[0]), max_value=int(sorted(AVERAGE_TRAN_CREDIT_AMT_options, reverse=True)[0]))
        # CNT_CREDIT_TRAN = st.slider('Select CNT_CREDIT_TRAN', min_value=int(sorted(CNT_CREDIT_TRAN_options)[0]), max_value=int(sorted(CNT_CREDIT_TRAN_options, reverse=True)[0]))
        # AVERAGE_TRAN_DEBIT_AMT = st.slider('Select AVERAGE_TRAN_DEBIT_AMT', min_value=int(sorted(AVERAGE_TRAN_DEBIT_AMT_options)[0]), max_value=int(sorted(AVERAGE_TRAN_DEBIT_AMT_options, reverse=True)[0]))
        # CNT_DEBIT_TRAN = st.slider('Select CNT_DEBIT_TRAN', min_value=int(sorted(CNT_DEBIT_TRAN_options)[0]), max_value=int(sorted(CNT_DEBIT_TRAN_options, reverse=True)[0]))
        # CREDIT_TO_DEBIT_RATIO = st.slider('Select CREDIT_TO_DEBIT_RATIO_options', min_value=int(sorted(CREDIT_TO_DEBIT_RATIO_options)[0]), max_value=int(sorted(CREDIT_TO_DEBIT_RATIO_options, reverse=True)[0]))
        # DAYS_SINCE_MOST_RECENT_TRAN = st.slider('Select DAYS_SINCE_MOST_RECENT_TRAN', min_value=int(sorted(DAYS_SINCE_MOST_RECENT_TRAN_options)[0]), max_value=int(sorted(DAYS_SINCE_MOST_RECENT_TRAN_options, reverse=True)[0]))


        # Dropdowns for categorical values
        TENURE = st.slider('Select TENURE', min_value=int(sorted(TENURE_options)[0]), max_value=int(sorted(TENURE_options, reverse=True)[0]))
        PREFERREDLOGINDEVICE = st.selectbox('Select PREFERREDLOGINDEVICE', PREFERREDLOGINDEVICE_options)
        CITYTIER = st.slider('Select CITYTIER', min_value=int(sorted(CITYTIER_options)[0]), max_value=int(sorted(CITYTIER_options, reverse=True)[0]))
        WAREHOUSETOHOME = st.slider('Select WAREHOUSETOHOME', min_value=int(sorted(WAREHOUSETOHOME_options)[0]), max_value=int(sorted(WAREHOUSETOHOME_options, reverse=True)[0]))
        PREFERREDPAYMENTMODE = st.selectbox('Select PREFERREDPAYMENTMODE', PREFERREDPAYMENTMODE_options)
        GENDER = st.selectbox('Select GENDER', GENDER_options)
        HOURSPENDONAPP = st.slider('Select HOURSPENDONAPP', min_value=int(sorted(HOURSPENDONAPP_options)[0]), max_value=int(sorted(HOURSPENDONAPP_options, reverse=True)[0]))
        NUMBEROFDEVICEREGISTERED = st.slider('Select NUMBEROFDEVICEREGISTERED', min_value=int(sorted(NUMBEROFDEVICEREGISTERED_options)[0]), max_value=int(sorted(NUMBEROFDEVICEREGISTERED_options, reverse=True)[0]))
        PREFEREDORDERCAT = st.selectbox('Select PREFEREDORDERCAT', PREFEREDORDERCAT_options)
        SATISFACTIONSCORE = st.slider('Select SATISFACTIONSCORE', min_value=int(sorted(SATISFACTIONSCORE_options)[0]), max_value=int(sorted(SATISFACTIONSCORE_options, reverse=True)[0]))
        MARITALSTATUS = st.selectbox('Select MARITALSTATUS', MARITALSTATUS_options)
        NUMBEROFADDRESS = st.slider('Select NUMBEROFADDRESS', min_value=int(sorted(NUMBEROFADDRESS_options)[0]), max_value=int(sorted(NUMBEROFADDRESS_options, reverse=True)[0]))
        COMPLAIN = st.slider('Select COMPLAIN', min_value=int(sorted(COMPLAIN_options)[0]), max_value=int(sorted(COMPLAIN_options, reverse=True)[0]))
        ORDERAMOUNTHIKEFROMLASTYEAR = st.slider('Select ORDERAMOUNTHIKEFROMLASTYEAR', min_value=int(sorted(ORDERAMOUNTHIKEFROMLASTYEAR_options)[0]), max_value=int(sorted(ORDERAMOUNTHIKEFROMLASTYEAR_options, reverse=True)[0]))
        COUPONUSED = st.slider('Select COUPONUSED', min_value=int(sorted(COUPONUSED_options)[0]), max_value=int(sorted(COUPONUSED_options, reverse=True)[0]))
        ORDERCOUNT = st.slider('Select ORDERCOUNT', min_value=int(sorted(ORDERCOUNT_options)[0]), max_value=int(sorted(ORDERCOUNT_options, reverse=True)[0]))
        DAYSINCELASTORDER = st.slider('Select DAYSINCELASTORDER', min_value=int(sorted(DAYSINCELASTORDER_options)[0]), max_value=int(sorted(DAYSINCELASTORDER_options, reverse=True)[0]))
        CASHBACKAMOUNT = st.slider('Select CASHBACKAMOUNT', min_value=int(sorted(CASHBACKAMOUNT_options)[0]), max_value=int(sorted(CASHBACKAMOUNT_options, reverse=True)[0]))

        # LOANAMOUNT = st.number_input("Select LOANAMOUNT greater than 0", value=0)
        
        if st.button("Predict"):
                prediction = session.sql(f"""
               SELECT CHURN_PREDICTION(
                    {TENURE},
                    '{PREFERREDLOGINDEVICE}',
					{CITYTIER},
					{WAREHOUSETOHOME},
					'{PREFERREDPAYMENTMODE}',
					'{GENDER}',
					{HOURSPENDONAPP},
					{NUMBEROFDEVICEREGISTERED},
					'{PREFEREDORDERCAT}',
					{SATISFACTIONSCORE},
					'{MARITALSTATUS}',
					{NUMBEROFADDRESS},
					{COMPLAIN},
					{ORDERAMOUNTHIKEFROMLASTYEAR},
					{COUPONUSED},
					{ORDERCOUNT},
					{DAYSINCELASTORDER},
					{CASHBACKAMOUNT}
                    
                )
            """).collect()

                # if prediction[0][0]=='0':
                #     st.write('Deal: Not Closed')
                # else:
                #     st.write('Deal: Closed')
                score = str(np.round((float(prediction[0][0])*100),decimals=2))
                st.write('**Probability of churn:**',score,'%')

                session.file.get('@"CUST_CHURN"."CUSTOMER"."MODEL_STAGE"/churn_model.pkl','/tmp')
                model = load_model('/tmp/churn_model')
        
                X = model[:-1].transform(df).drop(columns=["CHURN"], axis=1)
                X_featurenames = X.columns.tolist()
                explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                     feature_names=X_featurenames, 
                                     class_names=['Not Closed','Closed'],
                                     discretize_continuous = True,
                                     verbose=True, mode='classification')
                
                df_local = pd.DataFrame({
                'TENURE': [TENURE],
                'PREFERREDLOGINDEVICE': [PREFERREDLOGINDEVICE],
                'CITYTIER': [CITYTIER],
                'WAREHOUSETOHOME': [WAREHOUSETOHOME],
                'PREFERREDPAYMENTMODE': [PREFERREDPAYMENTMODE],
                'GENDER': [GENDER],
                'HOURSPENDONAPP': [HOURSPENDONAPP],
                'NUMBEROFDEVICEREGISTERED': [NUMBEROFDEVICEREGISTERED],
                'PREFEREDORDERCAT': [PREFEREDORDERCAT],
                'SATISFACTIONSCORE': [SATISFACTIONSCORE],
                'MARITALSTATUS': [MARITALSTATUS],
                'NUMBEROFADDRESS': [NUMBEROFADDRESS],
                'COMPLAIN': [COMPLAIN],
                'ORDERAMOUNTHIKEFROMLASTYEAR': [ORDERAMOUNTHIKEFROMLASTYEAR],
				'COUPONUSED': [COUPONUSED],
				'ORDERCOUNT': [ORDERCOUNT],
				'DAYSINCELASTORDER': [DAYSINCELASTORDER],
				'CASHBACKAMOUNT': [CASHBACKAMOUNT]})
                
                instance_to_predict = model[:-1].transform(df_local)
        
                exp = explainer.explain_instance(instance_to_predict.values.flatten(), model.named_steps["trained_model"].predict_proba,labels=(1,))
        
                st.subheader('Local Explanability of Model')
                exp.as_pyplot_figure()
                st.pyplot()
         



if __name__ == '__main__':
    dataframe = EDA_Dataframe_Analysis()
    info = Attribute_Information()
    # plot_charts = Charts()
    conn = st.connection("snowflake")
    session = conn.session()
    main()