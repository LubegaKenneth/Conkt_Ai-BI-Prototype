# app.py - AI Business Intelligence Prototype for Windows
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="📊",
    layout="wide"
)

# Simple function to analyze data (without OpenAI for now)
def analyze_data_simple(df):
    """Basic data analysis without AI (for testing)"""
    
    insights = []
    
    # Basic statistics
    insights.append(f"📊 Your dataset has {len(df)} rows and {len(df.columns)} columns")
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        insights.append(f"💰 Numeric fields found: {', '.join(numeric_cols)}")
        
        # Find highest values
        for col in numeric_cols[:2]:  # Check first 2 numeric columns
            max_val = df[col].max()
            max_idx = df[col].idxmax()
            insights.append(f"📈 Highest {col}: {max_val:,.0f}")
    
    # Find categorical data
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        insights.append(f"📋 Categories found: {', '.join(cat_cols[:3])}")
    
    return "\n".join([f"• {insight}" for insight in insights])

def create_simple_charts(df):
    """Create basic charts from data"""
    charts = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Chart 1: Bar chart of first categorical vs first numeric
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Group and sum
        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=grouped.index,
            y=grouped.values,
            title=f"{num_col} by {cat_col}",
            labels={'x': cat_col, 'y': num_col}
        )
        charts.append(("Bar Chart", fig))
    
    # Chart 2: Line chart if we can find a date-like column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue
    
    if date_col and numeric_cols:
        num_col = numeric_cols[0]
        df_sorted = df.sort_values(date_col)
        
        fig = px.line(
            df_sorted,
            x=date_col,
            y=num_col,
            title=f"{num_col} Over Time"
        )
        charts.append(("Time Series", fig))
    
    # Chart 3: Histogram of first numeric column
    if numeric_cols:
        num_col = numeric_cols[0]
        fig = px.histogram(
            df,
            x=num_col,
            title=f"Distribution of {num_col}"
        )
        charts.append(("Distribution", fig))
    
    return charts

def main():
    # Header
    st.title("🤖 AI Business Intelligence Prototype")
    st.markdown("Upload your CSV data and get instant business insights!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your business data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Show basic info
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            
            # Data overview
            st.header("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size} bytes")
            
            # Show preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10))
            
            # Basic analysis
            st.header("🧠 Quick Insights")
            insights = analyze_data_simple(df)
            st.info(insights)
            
            # Create charts
            st.header("📈 Visualizations")
            charts = create_simple_charts(df)
            
            if charts:
                for chart_name, fig in charts:
                    st.subheader(chart_name)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No suitable data found for charts. Make sure your CSV has numeric columns.")
            
            # Data summary
            st.header("📋 Statistical Summary")
            st.dataframe(df.describe())
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Make sure your file is a valid CSV with column headers in the first row.")
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        st.markdown("""
        ### What this app does:
        - 📊 Analyzes your business data instantly
        - 📈 Creates automatic charts and graphs  
        - 💡 Provides quick business insights
        - 📋 Shows statistical summaries
        
        ### Sample CSV format:
        ```
        date,product,revenue,quantity
        2024-01-01,Product A,1200,10
        2024-01-02,Product B,800,5
        ```
        """)

if __name__ == "__main__":
    main()
