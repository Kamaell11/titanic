import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import folium
from streamlit_folium import folium_static
from folium import Marker, PolyLine, Popup, Icon, CustomIcon

# Page config
st.set_page_config(page_title="Titanic Model", page_icon=":ship:", layout="wide")

# User welcome
st.title("Welcome on titanic board!")

# Load Titanic dataset as data frame
df = pd.read_csv("data/titanic.csv")

# <=== SIDEBAR ===>
st.sidebar.header(" Filters: ")

survived = st.sidebar.multiselect(
    "Select the survive option:",
    options=df["survived"].unique(),
    default=df["survived"].unique()
)

pclass = st.sidebar.multiselect(
    "Select the classes option:",
    options=df["pclass"].unique(),
    default=df["pclass"].unique()
)

sex = st.sidebar.multiselect(
    "Select the sex option:",
    options=df["sex"].unique(),
    default=df["sex"].unique()
)

df_selection = df.query("survived == @survived & pclass == @pclass & sex == @sex")

# Load dataframe with filter selection
st.dataframe(df_selection)

st.sidebar.title("Data processing options")
columns = list(df.columns)

selected_cols = st.sidebar.multiselect("Choose Columns", columns, default=columns)
if len(selected_cols) == 0:
    st.warning("No columns selected.")
else:
    process_method = st.sidebar.selectbox(
        "Choose a processing method",
        ["Delete rows with missing values", "Delete selected columns", "Fill in the missing mean values"]
    )
if st.sidebar.button("Perform the operation"):
    if process_method == "Delete rows with missing values":
        for col in selected_cols:
            if df[col].isna().any():
                df = df.drop(col, axis=1)
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
        st.success("The missing values were completed successfully!")
        st.dataframe(df)

# <=== MAINPAGE ===>
main_columns = st.columns((1, 2))

with main_columns[0]:
    st.subheader("Total Passagers: ")
    st.subheader(len(df_selection))
    st.subheader("Total Male: ")
    st.subheader(len(df_selection[df_selection["sex"] == "male"]))
    st.subheader("Total Female: ")
    st.subheader(len(df_selection[df_selection["sex"] == "female"]))

    missing_values = df.isnull().sum()
    total_cells = np.product(df.shape)
    total_missing = missing_values.sum()
    missing_prec = int((total_missing / total_cells) * 100) if total_cells != 0 else 0
    st.write('The number of empty columns in dataset: \n ')
    st.write(missing_values)
    st.write(f'Missing data is: {missing_prec}%')

with main_columns[1]:
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

    if 'sex' not in df.columns:
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

# <=== Machine Learning ===>

# Cleaning data
# fill_median
median = df['age'].median()
df['age'].fillna(median, inplace=True)
median = df['fare'].median()
df['fare'].fillna(median, inplace=True)
# most_common_value
most_common_value = df['embarked'].mode()[0]
df['embarked'].fillna(most_common_value, inplace=True)
# drop columns
df.drop('cabin', axis=1, inplace=True)
df.drop('boat', axis=1, inplace=True)
df.drop('body', axis=1, inplace=True)
df.drop('home_dest', axis=1, inplace=True)

# Chaning variables to binary
df['sex'] = pd.factorize(df['sex'])[0]
df['embarked'] = pd.factorize(df['embarked'])[0]

# Extraction of features and target variable
X = df.drop(['survived', 'name', 'ticket'], axis=1)
y = df['survived']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple machine learning models
models = {'Logistic Regression': LogisticRegression(), 
          'Random Forest': RandomForestClassifier(),
          'Decision Tree': DecisionTreeClassifier(), 
          'Support Vector Machine': SVC()}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    st.write(f"{name} accuracy: {accuracy}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
st.info(f"The best model {best_model} has been saved successfully!")

# <=== Prediction ===>

# Load the best model
best_model = joblib.load('best_model.pkl')

# Scale the features
scaler = StandardScaler()

# Get user input
age = st.number_input("Enter your age:", min_value=0, max_value=150, step=1)
sex = st.radio("Select your sex:", ['male', 'female'])
pclass = st.selectbox("Select your passenger class:", [1, 2, 3])
sibsp = st.number_input("Enter the number of siblings/spouses aboard:", min_value=0, max_value=10, step=1)
parch = st.number_input("Enter the number of parents/children aboard:", min_value=0, max_value=10, step=1)
fare = st.number_input("Enter the fare paid:", min_value=0.0, max_value=1000.0, step=1.0)
embarked = st.selectbox("Select the port of embarkation:", ['C', 'Q', 'S'])

st.info("Cherbourg (C) - the port in France, Queenstown (Q) - the port in Ireland, Southampton (S) - the port in England.")
# Coordinates for ports
port_coordinates = {
    'Cherbourg': (49.633, -1.617),
    'Queenstown': (51.849, -8.297),
    'Southampton': (50.909, -1.404)
}

# Titanic route (adjust as needed)
titanic_route = [
    port_coordinates['Southampton'],
    (41.7325, -49.9469),  # Titanic sinking location
    port_coordinates['Cherbourg']
]

# Coordinates of the Titanic sinking location
sink_coordinates = (41.7325, -49.9469)
# Creating a map
mymap = folium.Map(location=(51.5074, -0.1278), zoom_start=3)

# Dodawanie markerów dla każdego portu
for port, coordinates in port_coordinates.items():
    Marker(location=coordinates, popup=port).add_to(mymap)

# Dodawanie przerywanej linii dla trasy Titanica
PolyLine(titanic_route, color="red", weight=2.5, opacity=1, dash_array="10").add_to(mymap)

# Dodawanie obrazka bezpośrednio na mapie w miejscu zatonięcia Titanica
image_path = "data/titanic.png"
icon = CustomIcon(image_path, icon_size=(50, 50))
popup_content = "Titanic's Wreck"
popup = Popup(popup_content, max_width=400)
folium.Marker(location=sink_coordinates, icon=icon, popup=popup).add_to(mymap)

# Utwórz dwie kolumny
map_column, video_column = st.columns(2)

# Wyświetlenie mapy w jednej kolumnie
with map_column:
    st.header("Map")
    folium_static(mymap)

# Wyświetlenie filmu YouTube w drugiej kolumnie
with video_column:
    st.header("Song about Titanic")
    video_url = "https://www.youtube.com/watch?v=FhL0hCGgMj4"
    st.video(video_url)



# Preprocess the input data
data = {'age': age, 'sex': sex, 'pclass': pclass, 'sibsp': sibsp, 'parch': parch, 'fare': fare, 'embarked': embarked}
df = pd.DataFrame(data, index=[0])
df['sex'] = pd.factorize(df['sex'])[0]
df['embarked'] = pd.factorize(df['embarked'])[0]
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df = scaler.fit_transform(df)

# Make the prediction
prediction = best_model.predict(df)
if st.sidebar.button("See Your chances to survive"):
    if prediction[0] == 0:
        st.write("Sorry, you did not survive the Titanic disaster.")
        st.write(f"Your chances of survival: {round(best_model.predict_proba(df)[0][1] * 100, 2)}%")
    else:
        st.write("Congratulations, you survived the Titanic disaster!")
        st.write(f"Your chances of survival: {round(best_model.predict_proba(df)[0][1] * 100, 2)}%")

