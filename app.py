import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Titanic Dataset Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance app appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #0277BD;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .footer {
        font-size: 0.8rem;
        color: #555;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    # Convert categorical features
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    # Extract title from name - use r prefix for raw string to avoid escape sequence warning
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Fix age data type
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Family size
    df['FamilySize'] = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard'] + 1
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    return df

# Load the dataset
df = load_data()

# Sidebar filters
st.sidebar.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

# Passenger class filter
pclass = st.sidebar.multiselect(
    'Passenger Class',
    options=sorted(df['Pclass'].unique()),
    default=sorted(df['Pclass'].unique())
)

# Gender filter
gender = st.sidebar.multiselect(
    'Gender',
    options=sorted(df['Sex'].unique()),
    default=sorted(df['Sex'].unique())
)

# Age range filter
age_range = st.sidebar.slider(
    'Age Range',
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max())),
    step=1
)

# Apply filters
filtered_df = df[
    (df['Pclass'].isin(pclass)) &
    (df['Sex'].isin(gender)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# Sidebar information
with st.sidebar.expander("About the Dataset", expanded=True):
    st.write("""
    The **Titanic dataset** contains information about the passengers aboard the RMS Titanic, 
    which sank on its maiden voyage in April 1912 after colliding with an iceberg.
    This dataset is often used for introducing machine learning concepts.
    """)

# Main content
st.markdown('<div class="main-header">Titanic Dataset Explorer</div>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Titanic Dataset Explorer! This application showcases the capabilities of Streamlit 
for creating interactive data visualizations and analyses. Use the sidebar filters to explore 
different subsets of the data and the tabs below to view different analyses.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Survival Analysis", "Passenger Demographics", "Relationships"])

with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="subsection-header">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Download button for the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="titanic_filtered.csv",
            mime="text/csv",
        )
    
    with col2:
        st.markdown('<div class="subsection-header">Dataset Statistics</div>', unsafe_allow_html=True)
        st.metric("Total Passengers", df.shape[0])
        st.metric("Filtered Passengers", filtered_df.shape[0])
        st.metric("Survival Rate", f"{filtered_df['Survived'].mean():.2%}")
    
    # Dataset description
    with st.expander("View Detailed Statistics", expanded=True):
        st.write(filtered_df.describe())
    
    # Missing values analysis
    with st.expander("Missing Values Analysis", expanded=True):
        missing_values = df.isnull().sum()
        st.write("Missing values per column:")
        st.write(missing_values[missing_values > 0])
        
        # Visualize missing values
        if any(missing_values > 0):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
            st.pyplot(fig)

with tab2:
    st.markdown('<div class="section-header">Survival Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Survival by Passenger Class</div>', unsafe_allow_html=True)
        fig = px.bar(
            filtered_df.groupby('Pclass')['Survived'].mean().reset_index(),
            x='Pclass',
            y='Survived',
            color='Pclass',
            labels={'Survived': 'Survival Rate', 'Pclass': 'Passenger Class'},
            text_auto='.2%',
            title='Survival Rate by Passenger Class'
        )
        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Survival by Gender</div>', unsafe_allow_html=True)
        fig = px.bar(
            filtered_df.groupby('Sex')['Survived'].mean().reset_index(),
            x='Sex',
            y='Survived',
            color='Sex',
            labels={'Survived': 'Survival Rate', 'Sex': 'Gender'},
            text_auto='.2%',
            title='Survival Rate by Gender'
        )
        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Survival by Age Group</div>', unsafe_allow_html=True)
        # Create age groups
        filtered_df['AgeGroup'] = pd.cut(
            filtered_df['Age'], 
            bins=[0, 12, 18, 30, 50, 100], 
            labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 'Adult (31-50)', 'Senior (51+)']
        )
        
        # Calculate survival rate by age group
        age_survival = filtered_df.groupby('AgeGroup', observed=True)['Survived'].mean().reset_index()
        
        fig = px.bar(
            age_survival,
            x='AgeGroup',
            y='Survived',
            color='AgeGroup',
            labels={'Survived': 'Survival Rate', 'AgeGroup': 'Age Group'},
            text_auto='.2%',
            title='Survival Rate by Age Group'
        )
        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Survival by Family Size</div>', unsafe_allow_html=True)
        # Group family sizes
        filtered_df['FamilySizeGroup'] = pd.cut(
            filtered_df['FamilySize'], 
            bins=[0, 1, 3, 6, 12], 
            labels=['Alone', 'Small (2-3)', 'Medium (4-6)', 'Large (7+)']
        )
        
        # Calculate survival rate by family size group
        family_survival = filtered_df.groupby('FamilySizeGroup', observed=True)['Survived'].mean().reset_index()
        
        fig = px.bar(
            family_survival,
            x='FamilySizeGroup',
            y='Survived',
            color='FamilySizeGroup',
            labels={'Survived': 'Survival Rate', 'FamilySizeGroup': 'Family Size'},
            text_auto='.2%',
            title='Survival Rate by Family Size'
        )
        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Interactive survival analysis
    st.markdown('<div class="subsection-header">Interactive Survival Analysis</div>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select analysis dimension:",
        ["Class & Gender", "Class & Age", "Gender & Age"]
    )
    
    if analysis_type == "Class & Gender":
        # Prepare data - add observed=True to avoid FutureWarning
        pivot_data = filtered_df.pivot_table(
            index='Pclass',
            columns='Sex',
            values='Survived',
            aggfunc='mean',
            observed=True
        ).fillna(0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2%', linewidths=.5, ax=ax)
        plt.title('Survival Rate by Class and Gender')
        st.pyplot(fig)
        
    elif analysis_type == "Class & Age":
        # Create age groups for this analysis
        temp_df = filtered_df.copy()
        temp_df['AgeGroup'] = pd.cut(
            temp_df['Age'], 
            bins=[0, 12, 18, 30, 50, 100], 
            labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 'Adult (31-50)', 'Senior (51+)']
        )
        
        # Prepare data - add observed=True to avoid FutureWarning
        pivot_data = temp_df.pivot_table(
            index='Pclass',
            columns='AgeGroup',
            values='Survived',
            aggfunc='mean',
            observed=True
        ).fillna(0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2%', linewidths=.5, ax=ax)
        plt.title('Survival Rate by Class and Age Group')
        st.pyplot(fig)
        
    elif analysis_type == "Gender & Age":
        # Create age groups for this analysis
        temp_df = filtered_df.copy()
        temp_df['AgeGroup'] = pd.cut(
            temp_df['Age'], 
            bins=[0, 12, 18, 30, 50, 100], 
            labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 'Adult (31-50)', 'Senior (51+)']
        )
        
        # Prepare data - add observed=True to avoid FutureWarning
        pivot_data = temp_df.pivot_table(
            index='Sex',
            columns='AgeGroup',
            values='Survived',
            aggfunc='mean',
            observed=True
        ).fillna(0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2%', linewidths=.5, ax=ax)
        plt.title('Survival Rate by Gender and Age Group')
        st.pyplot(fig)

with tab3:
    st.markdown('<div class="section-header">Passenger Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Age Distribution</div>', unsafe_allow_html=True)
        # Show histogram with kernel density estimate
        fig = px.histogram(
            filtered_df, 
            x='Age',
            color='Survived',
            marginal='violin',
            labels={'Age': 'Age (years)', 'count': 'Number of Passengers'},
            title='Age Distribution with Survival Indication',
            color_discrete_map={0: '#EF553B', 1: '#00CC96'},
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Passenger Class Distribution</div>', unsafe_allow_html=True)
        # Create a proper dataframe for the pie chart to avoid length mismatch error
        pclass_counts = filtered_df['Pclass'].value_counts().reset_index()
        pclass_counts.columns = ['Pclass', 'Count']
        
        fig = px.pie(
            pclass_counts,
            values='Count',
            names='Pclass',
            title='Passenger Class Distribution',
            color='Pclass',
            color_discrete_map={1: '#636EFA', 2: '#EF553B', 3: '#00CC96'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Gender Distribution</div>', unsafe_allow_html=True)
        # Create a proper dataframe for the pie chart to avoid length mismatch error
        sex_counts = filtered_df['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        
        fig = px.pie(
            sex_counts,
            values='Count',
            names='Sex',
            title='Gender Distribution',
            color='Sex',
            color_discrete_map={'male': '#636EFA', 'female': '#EF553B'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Fare Distribution</div>', unsafe_allow_html=True)
        # Create a box plot for fare distribution by passenger class
        fig = px.box(
            filtered_df,
            x='Pclass',
            y='Fare',
            color='Pclass',
            labels={'Pclass': 'Passenger Class', 'Fare': 'Fare (£)'},
            title='Fare Distribution by Passenger Class',
            points='all'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive demographic analysis with Altair
    st.markdown('<div class="subsection-header">Interactive Demographic Chart</div>', unsafe_allow_html=True)
    
    # Create interactive scatter plot with Altair
    chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x=alt.X('Age:Q', title='Age'),
        y=alt.Y('Fare:Q', title='Fare (£)'),
        color=alt.Color('Sex:N', title='Gender'),
        size=alt.Size('FamilySize:Q', title='Family Size'),
        shape=alt.Shape('Pclass:N', title='Passenger Class'),
        tooltip=['Name', 'Age', 'Sex', 'Fare', 'Pclass', 'Survived']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Relationships & Correlations</div>', unsafe_allow_html=True)
    
    # Create a correlation matrix
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    # Display the correlation matrix as a heatmap
    fig = px.imshow(
        correlation,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix of Numeric Features'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationship analysis
    st.markdown('<div class="subsection-header">Feature Relationships</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs. Fare with survival indicator
        fig = px.scatter(
            filtered_df,
            x='Age',
            y='Fare',
            color='Survived',
            size='FamilySize',
            hover_name='Name',
            labels={'Age': 'Age (years)', 'Fare': 'Fare (£)', 'Survived': 'Survived'},
            title='Age vs. Fare by Survival Status',
            color_discrete_map={0: '#EF553B', 1: '#00CC96'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Family size vs. Survival with passenger class
        fig = px.scatter(
            filtered_df,
            x='FamilySize',
            y='Survived',
            color='Pclass',
            facet_col='Sex',
            hover_name='Name',
            labels={'FamilySize': 'Family Size', 'Survived': 'Survived', 'Pclass': 'Passenger Class'},
            title='Family Size vs. Survival by Gender and Class',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fare per person analysis
    st.markdown('<div class="subsection-header">Fare Per Person Analysis</div>', unsafe_allow_html=True)
    
    # Group fare per person into bins
    filtered_df['FarePerPersonBin'] = pd.qcut(
        filtered_df['FarePerPerson'].clip(upper=100),
        4,
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Calculate survival rate by fare per person bin
    fare_survival = filtered_df.groupby('FarePerPersonBin', observed=True)['Survived'].mean().reset_index()
    
    fig = px.bar(
        fare_survival,
        x='FarePerPersonBin',
        y='Survived',
        color='FarePerPersonBin',
        labels={'Survived': 'Survival Rate', 'FarePerPersonBin': 'Fare Per Person'},
        text_auto='.2%',
        title='Survival Rate by Fare Per Person'
    )
    fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Created with Streamlit to demonstrate data visualization capabilities.</p>
    <p>Data source: Titanic Dataset</p>
</div>
""", unsafe_allow_html=True)
