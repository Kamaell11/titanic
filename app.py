import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
# Page config
st.set_page_config(page_title="Titanic Model",
                   page_icon=":ship:",
                   layout="wide")

# User welcome
st.title("Welcome on titanic board!")


# Load Titanic dataset as data frame
df = pd.read_csv("data/titanic.csv")




# <=== SIDEBAR ===>
st.sidebar.header(" Filters: ")

survived = st.sidebar.multiselect(
    "Select the survive option:",
    options= df["survived"].unique(),
    default= df["survived"].unique()
)

pclass = st.sidebar.multiselect(
    "Select the classes option:",
    options= df["pclass"].unique(),
    default= df["pclass"].unique()
)

sex = st.sidebar.multiselect(
    "Select the sex option:",
    options= df["sex"].unique(),
    default= df["sex"].unique()
)

df_selection = df.query(
    "survived == @survived & pclass == @pclass & sex == @sex"
    

    
)

# Load dataframe with filter selection
st.dataframe(df_selection)

st.sidebar.title("Data processing options")
columns = list(df.columns)

selected_cols = st.sidebar.multiselect("Chose Columncs", columns, default=columns)
if len(selected_cols) == 0:
    st.warning("No columns selected.")
else:
    process_method = st.sidebar.selectbox("Choose a processing method", ["Delete rows with missing values", "Delete selected columns",  "Fill in the missing mean values"])
if st.sidebar.button("Perform the operation"):
    if process_method == "Delete rows with missing values":
        for col in selected_cols:
            if df[col].isna().any():
                df = df.drop(col, axis=1)
                # st.warning(f"Column '{col}' was removed due to missing values")
            # else:
                # st.success(f"No missing values ​​were found in the column'{col}'")
        st.success("The selected columns have been deleted successfully!")
        st.dataframe(df)
    elif process_method == "Delete selected columns":
        for col in selected_cols:
            df.drop(col, axis=1, inplace=True)
        st.success("The selected columns have been deleted successfully!")
        st.dataframe(df)
    elif process_method == "Fill in the missing mean values":
        df_copy = df.copy()
        for col in selected_cols:
            col_type = df_copy[col].dtype
            if col_type == "object":
                df_copy[col].fillna(df_copy[col].value_counts().index[0], inplace=True)
            else:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        df = df_copy
        st.success("The missing values ​​were completed successfully!")
        st.dataframe(df)





# <=== MAINPAGE ===>
st.title(":ship: Titanic Model")
st.markdown("##")

total_psg = len(df_selection)
total_men = len(df_selection[df_selection["sex"] == "male"])

total_fem = len(df_selection[df_selection["sex"] == "female"])

left_column, middle_column, right_column = st.columns((1,3,1))


with left_column:
    st.subheader("Total Passagers: ")
    st.subheader(total_psg)
    st.subheader("Total Male: ")
    st.subheader(total_men)
    st.subheader("Total Female: ")
    st.subheader(total_fem)

with middle_column:
    if 'age' not in df.columns:
        st.warning("The column 'age' does not exist in the dataframe")
    else:

        # Create an interactive element in the user interface that allows you to select a gender
        sex = st.selectbox("Select gender", options=["All", "male", "female"])

        # Create an interactive element in the user interface that allows you to select an age range
        age_range = st.slider("Select age range", min_value=0, max_value=100, value=(0, 100))

        # Selection of relevant records based on selected options
        if sex == "All":
            df_selection = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
        else:
            df_selection = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) & (df["sex"] == sex)]

        # Calculation of the number of passengers based on the selected options
        total_psg = len(df_selection)

        # Calculation of the number of male passengers based on the selected options
        total_men = len(df_selection[df_selection["sex"] == "male"])

        # Calculation of the number of female passengers based on the selected options
        total_women = len(df_selection[df_selection["sex"] == "female"])

        # Creating a DataFrame containing values ​​for the chart
        data = pd.DataFrame({"Gender": ["male", "female"], "Count": [total_men, total_women]})

        # Generating a bar chart using the Plotly library
        fig = px.bar(data, x="Gender", y="Count")

        # Adding a title to the chart
        fig.update_layout(title="Passenger gender distribution",
                        height=400, 
                        autosize=True)



        # Displaying the graph in Streamlit
        st.plotly_chart(fig)

    if 'sex'  not in df.columns:
        st.warning("The column 'sex or age' does not exist in the dataframe")
    else:

            # Displaying an interactive graph
        fig = go.Figure()

        # Chances of survival for men
        male_survival_rate = df[df['sex'] == 'male'].groupby('pclass')['survived'].mean()
        fig.add_trace(go.Bar(x=male_survival_rate.index, y=male_survival_rate.values, name='men'))

        # Chances of survival for women
        female_survival_rate = df[df['sex'] == 'female'].groupby('pclass')['survived'].mean()
        fig.add_trace(go.Bar(x=female_survival_rate.index, y=female_survival_rate.values, name='female'))

        # Added x and y axis labels and chart title
        fig.update_layout(title='Titanic Survival Rate by Class and Gender',
                        xaxis_title='Class',
                        yaxis_title='Survival Rate',
                        autosize=True)

        

        # Viewing the graph in the Streamlit app
        st.plotly_chart(fig)

with right_column:
    # Check missing values
    missing_values = df.isnull().sum()

    total_cells = np.product(df.shape)
    total_missing = missing_values.sum()

    # percent of data that is missing
    if total_cells == 0:
        missing_prec = 0
    else:
        missing_prec = int((total_missing/total_cells) * 100)


    st.write(f'The number of empty columns in dataset: \n ')
    
    st.write(missing_values)
    st.write(f'Missing data is: {missing_prec}%')
    

    