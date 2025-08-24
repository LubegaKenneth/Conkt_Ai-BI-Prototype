import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import json
from datetime import datetime, timedelta
import re

# Configure Streamlit page
st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .overview-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem;
        text-align: center;
    }
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-ai {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .quick-insight {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .ai-thinking {
        background: #fef3c7;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'auto_insights' not in st.session_state:
    st.session_state.auto_insights = []
if 'mock_data_generated' not in st.session_state:
    st.session_state.mock_data_generated = False

# Mock Data Generation (like the React version)
def generate_mock_data():
    """Generate sample business data for demo purposes"""
    np.random.seed(42)
    
    # Sample sales data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    revenue = [65000, 78000, 82000, 91000, 88000, 95000]
    orders = [120, 140, 155, 170, 165, 180]
    customers = [89, 102, 118, 134, 128, 145]
    
    sales_data = pd.DataFrame({
        'Month': months,
        'Revenue': revenue,
        'Orders': orders,
        'Customers': customers
    })
    
    # Regional data
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    regional_revenue = [180000, 145000, 120000, 65000, 45000]
    growth_rates = [12.5, 8.3, 15.2, 6.7, 9.1]
    
    regional_data = pd.DataFrame({
        'Region': regions,
        'Revenue': regional_revenue,
        'Growth_Rate': growth_rates
    })
    
    # Product data
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    product_sales = [45000, 32000, 28000, 18000, 12000]
    market_share = [35, 25, 20, 12, 8]
    
    product_data = pd.DataFrame({
        'Product': products,
        'Sales': product_sales,
        'Market_Share': market_share
    })
    
    return sales_data, regional_data, product_data

# Simulated OpenAI API Integration (Ready for real API)
def simulate_openai_response(query, data_context=""):
    """
    Simulated OpenAI API call - Ready to replace with real OpenAI API
    Replace this function with actual OpenAI API calls when ready
    """
    time.sleep(1.5)  # Simulate API delay
    
    query_lower = query.lower()
    
    # Enhanced response patterns
    responses = {
        'revenue trends': f"Based on the comprehensive data analysis, revenue demonstrates a robust upward trajectory with a 46% increase from January to June. The growth pattern shows consistent momentum with only minor fluctuations in May. Q2 performance exceeded projections by 12%, indicating strong market positioning and effective sales strategies. Key drivers include increased customer acquisition and higher average order values.",
        
        'top products': f"Product performance analysis reveals Product A as the market leader with 35% market share generating $45K in sales revenue. This represents a commanding position driven by superior customer satisfaction scores and 78% repeat purchase rates. Product B follows with solid 25% market share, while the long tail of Products C-E collectively represents 40% of revenue, suggesting a balanced portfolio with opportunities for optimization.",
        
        'regional performance': f"Geographic analysis shows Asia Pacific as the standout performer with 15.2% growth rate, significantly outpacing other regions. This growth is attributed to market expansion initiatives and localized product offerings. North America maintains strong performance at 12.5% growth, while Europe shows stability with steady 8.3% growth. Emerging markets (Latin America, Middle East) present untapped potential with infrastructure investments recommended.",
        
        'customer insights': f"Customer analytics reveal exceptional growth with 63% year-over-year acquisition increase. Average customer lifetime value has improved by 18% to $847, while retention rates peak at 89% in the premium segment. Customer acquisition cost has decreased by 12% due to improved targeting and referral programs. Behavioral analysis suggests cross-selling opportunities could increase revenue per customer by additional 25%.",
        
        'sales forecast': f"Predictive modeling based on historical trends, seasonal patterns, and market indicators projects Q3 monthly revenue of $105-110K with 87% confidence interval. Holiday seasonality analysis suggests Q4 could experience 25-30% uplift, potentially reaching $130-140K monthly peaks. Risk factors include market saturation in mature segments, offset by expansion opportunities in emerging markets.",
        
        'data quality': f"Data integrity analysis shows {len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 'sample'} records with 94% completeness rate. Identified outliers represent 3% of dataset, likely indicating premium transactions rather than data errors. Recommend implementing automated data validation pipelines and real-time quality monitoring for enhanced analytics reliability.",
        
        'market analysis': f"Competitive positioning analysis indicates strong market share growth in core segments. Brand sentiment analysis (simulated) shows 78% positive customer feedback with 'product quality' and 'customer service' as top differentiators. Market opportunity sizing suggests 40% headroom in current segments with expansion potential worth $2.3M annually.",
        
        'operational efficiency': f"Operational metrics analysis reveals 23% improvement in order fulfillment speed and 18% reduction in customer service response times. Supply chain optimization has decreased costs by 11% while maintaining 99.2% order accuracy. Recommend investing in automation technologies to scale current efficiency gains.",
        
        'default': f"I've analyzed your business data comprehensively and can provide insights on revenue optimization, customer behavior, market trends, operational efficiency, and growth opportunities. My analysis capabilities include predictive modeling, trend analysis, anomaly detection, and strategic recommendations. What specific area would you like me to deep-dive into?"
    }
    
    # Pattern matching with context
    for key, response in responses.items():
        if key in query_lower:
            return response
    
    # If no specific pattern matches, return contextual default
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        return f"I can analyze your {len(df)} records across {len(df.columns)} dimensions. Your data contains valuable insights about trends, performance, and opportunities. Try asking about specific metrics like revenue trends, top performers, customer analysis, or forecasting. I'm ready to provide detailed business intelligence insights!"
    
    return responses['default']

# Enhanced Utility Functions
def detect_column_types(df):
    """Enhanced column type detection"""
    if df.empty:
        raise ValueError("Dataset is empty. Please upload valid data.")
    numeric_columns = []
    categorical_columns = []
    
    for col in df.columns:
        # Check if column is already numeric
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_columns.append(col)
        else:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
            except:
                categorical_columns.append(col)
    
    if not numeric_columns and not categorical_columns:
        raise ValueError("No valid columns detected in the dataset.")
    return numeric_columns, categorical_columns

def detect_date_columns(df):
    """Enhanced date column detection"""
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
        elif any(word in col.lower() for word in ['date', 'time', 'month', 'year', 'day']):
            try:
                pd.to_datetime(df[col].head(), errors='raise')
                date_columns.append(col)
            except:
                continue
    return date_columns

@st.cache_data
def calculate_advanced_metrics(df, numeric_cols):
    """Calculate advanced business metrics"""
    metrics = {}
    
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) > 0:
            metrics[col] = {
                'total': values.sum(),
                'average': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'trend': calculate_trend(values),
                'growth_rate': calculate_growth_rate(values),
                'volatility': calculate_volatility(values)
            }
    
    return metrics

def calculate_trend(values):
    """Enhanced trend calculation"""
    if len(values) < 2:
        return 0
    
    # Linear regression for trend
    x = np.arange(len(values))
    coefficients = np.polyfit(x, values, 1)
    slope = coefficients[0]
    
    # Convert to percentage
    mean_val = np.mean(values)
    if mean_val != 0:
        return (slope / mean_val) * 100
    return 0

def calculate_growth_rate(values):
    """Calculate period-over-period growth rate"""
    if len(values) < 2:
        return 0
    
    first_val = values.iloc[0] if hasattr(values, 'iloc') else values[0]
    last_val = values.iloc[-1] if hasattr(values, 'iloc') else values[-1]
    
    if first_val != 0:
        return ((last_val - first_val) / first_val) * 100
    return 0

def calculate_volatility(values):
    """Calculate volatility/stability measure"""
    if len(values) < 2:
        return 0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if mean_val != 0:
        return (std_val / mean_val) * 100  # Coefficient of variation
    return 0

@st.cache_data
def generate_comprehensive_insights(df):
    """Generate comprehensive AI insights"""
    insights = []
    numeric_cols, categorical_cols = detect_column_types(df)
    
    # Advanced metrics calculation
    metrics = calculate_advanced_metrics(df, numeric_cols)
    
    # 1. Data Overview Insight
    insights.append({
        'type': 'overview',
        'title': '📊 Comprehensive Data Overview',
        'insight': f'Your dataset contains {len(df):,} records with {len(numeric_cols)} quantitative metrics and {len(categorical_cols)} categorical dimensions. Data completeness: {(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%. This represents a {"robust" if len(df) > 1000 else "moderate"} dataset suitable for {"advanced" if len(numeric_cols) > 3 else "basic"} analytics.',
        'priority': 'high'
    })
    
    # 2. Revenue/Performance Insight
    if numeric_cols:
        primary_metric = numeric_cols[0]
        if primary_metric in metrics:
            metric_data = metrics[primary_metric]
            trend_desc = "strong upward" if metric_data['trend'] > 10 else "upward" if metric_data['trend'] > 0 else "declining"
            
            insights.append({
                'type': 'performance',
                'title': f'💰 {primary_metric} Performance Analysis',
                'insight': f'{primary_metric} shows {trend_desc} momentum with {metric_data["trend"]:.1f}% trend coefficient. Total value: {metric_data["total"]:,.2f}, averaging {metric_data["average"]:.2f} per record. Volatility index: {metric_data["volatility"]:.1f}% indicating {"high variance" if metric_data["volatility"] > 30 else "stable performance"}. Growth rate: {metric_data["growth_rate"]:.1f}% over the analysis period.',
                'priority': 'high'
            })
    
    # 3. Market Segmentation Insight
    if categorical_cols and numeric_cols:
        segment_col = categorical_cols[0]
        value_col = numeric_cols[0]
        
        segment_analysis = df.groupby(segment_col)[value_col].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        top_segment = segment_analysis.index[0]
        segment_concentration = (segment_analysis.loc[top_segment, 'sum'] / segment_analysis['sum'].sum()) * 100
        
        insights.append({
            'type': 'segmentation',
            'title': f'🎯 Market Segmentation Intelligence',
            'insight': f'"{top_segment}" dominates your {segment_col} portfolio with {segment_concentration:.1f}% concentration and {segment_analysis.loc[top_segment, "mean"]:.2f} average performance. Portfolio diversification index: {len(segment_analysis)}/{len(df)} unique segments. {"High concentration risk" if segment_concentration > 40 else "Balanced portfolio distribution"} detected. Top 3 segments account for {(segment_analysis.head(3)["sum"].sum() / segment_analysis["sum"].sum() * 100):.1f}% of total value.',
            'priority': 'medium'
        })
    
    # 4. Anomaly Detection Insight
    if numeric_cols:
        anomaly_col = numeric_cols[0]
        values = pd.to_numeric(df[anomaly_col], errors='coerce').dropna()
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
        
        if len(outliers) > 0:
            insights.append({
                'type': 'anomaly',
                'title': '🚨 Anomaly Detection Alert',
                'insight': f'Identified {len(outliers)} statistical outliers in {anomaly_col} ({(len(outliers)/len(values)*100):.1f}% of data). Outlier range: {outliers.min():.2f} to {outliers.max():.2f} vs normal range {Q1:.2f}-{Q3:.2f}. These anomalies may represent {"premium transactions" if outliers.mean() > values.mean() else "data quality issues"} requiring {"strategic focus" if outliers.mean() > values.mean() else "investigation"}.',
                'priority': 'high'
            })
    
    # 5. Predictive Insight
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]
        correlation = df[col1].corr(df[col2])
        
        insights.append({
            'type': 'predictive',
            'title': f'🔮 Predictive Analytics Intelligence',
            'insight': f'{col1} and {col2} show {abs(correlation):.2f} correlation coefficient, indicating {"strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"} relationship. Predictive power: {"High" if abs(correlation) > 0.7 else "Medium" if abs(correlation) > 0.4 else "Low"}. This {"enables" if abs(correlation) > 0.5 else "limits"} forecasting capabilities. Recommendation: {"Leverage this relationship for predictive modeling" if abs(correlation) > 0.5 else "Explore additional variables for better prediction accuracy"}.',
            'priority': 'medium'
        })
    
    return sorted(insights, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)[:6]

def create_advanced_visualizations(df, chart_type, cols):
    """Create advanced visualizations with multiple chart types"""
    numeric_cols, categorical_cols = detect_column_types(df)
    
    if chart_type == 'overview_dashboard':
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trends', 'Performance Distribution', 'Category Breakdown', 'Growth Analysis'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        if numeric_cols:
            # Line chart
            fig.add_trace(
                go.Scatter(x=df.index, y=df[numeric_cols[0]], name=numeric_cols[0], line=dict(width=3)),
                row=1, col=1
            )
            
            # Bar chart
            if categorical_cols:
                grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().head(5)
                fig.add_trace(
                    go.Bar(x=grouped.index, y=grouped.values, name="Top Categories"),
                    row=1, col=2
                )
                
                # Pie chart
                fig.add_trace(
                    go.Pie(labels=grouped.index, values=grouped.values, name="Distribution"),
                    row=2, col=1
                )
        
        fig.update_layout(height=800, title_text="Comprehensive Business Intelligence Dashboard")
        return fig
    
    elif chart_type == 'trend_analysis':
        if numeric_cols:
            fig = go.Figure()
            
            # Primary metric
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[numeric_cols[0]],
                mode='lines+markers',
                name=numeric_cols[0],
                line=dict(width=3, color='#1f77b4')
            ))
            
            # Trend line
            z = np.polyfit(range(len(df)), df[numeric_cols[0]], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=p(range(len(df))),
                mode='lines',
                name='Trend Line',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title=f'{numeric_cols[0]} Advanced Trend Analysis',
                xaxis_title='Period',
                yaxis_title=numeric_cols[0]
            )
            return fig
    
    return None

def process_advanced_nlp_query(query, df):
    """Enhanced NLP query processing with OpenAI simulation"""
    query_lower = query.lower()
    numeric_cols, categorical_cols = detect_column_types(df)
    
    try:
        # Simulate OpenAI API call
        with st.spinner("🧠 Processing with AI intelligence..."):
            ai_response = simulate_openai_response(query)
        
        # Generate appropriate visualization based on query
        chart = None
        
        if any(word in query_lower for word in ['trend', 'over time', 'time series', 'forecast']):
            chart = create_advanced_visualizations(df, 'trend_analysis', numeric_cols)
        elif any(word in query_lower for word in ['dashboard', 'overview', 'summary']):
            chart = create_advanced_visualizations(df, 'overview_dashboard', None)
        elif any(word in query_lower for word in ['compare', 'comparison', 'vs']):
            if len(numeric_cols) >= 2:
                chart = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                 title=f'{numeric_cols[0]} vs {numeric_cols[1]} Analysis',
                                 trendline="ols")
        elif any(word in query_lower for word in ['top', 'best', 'highest', 'ranking']):
            if numeric_cols and categorical_cols:
                grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().sort_values(ascending=False).head(8)
                chart = px.bar(x=grouped.index, y=grouped.values,
                             title=f'Top {categorical_cols[0]} Performance',
                             labels={'x': categorical_cols[0], 'y': numeric_cols[0]})
        
        return ai_response, chart
        
    except Exception as e:
        return f"I encountered an issue processing that query. Error: {str(e)}", None

def validate_data(df):
    """Validate data for issues"""
    issues = []
    if df.duplicated().sum() > 0:
        issues.append(f"Found {df.duplicated().sum()} duplicate rows")
    if df.isna().sum().sum() > len(df) * 0.5:
        issues.append("High missing value rate (>50%)")
    return issues

# Main Application
def main():
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem; border-radius: 10px;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🧠 AI Business Intelligence</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Advanced Analytics • Natural Language Queries • AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        tab = st.radio(
            "Select Module:",
            ["📁 Data Upload", "📊 Overview Dashboard", "🔍 Advanced Analytics", "🤖 AI Assistant"],
            help="Navigate between different analysis modules"
        )
        
        # Data status
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            numeric_cols, categorical_cols = detect_column_types(df)
            
            st.markdown("### ✅ Data Status")
            st.success(f"**{len(df):,}** records loaded")
            st.info(f"**{len(numeric_cols)}** numeric columns")
            st.info(f"**{len(categorical_cols)}** categorical columns")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Refresh Insights"):
                st.session_state.auto_insights = generate_comprehensive_insights(df)
                st.success("Insights refreshed!")
        
        # Demo data option
        st.markdown("### 🎲 Demo Mode")
        if st.button("Load Sample Data"):
            sales_data, regional_data, product_data = generate_mock_data()
            # Combine all sample data
            combined_data = pd.concat([
                sales_data.assign(Type='Sales'),
                regional_data.assign(Type='Regional'), 
                product_data.assign(Type='Product')
            ], ignore_index=True)
            
            st.session_state.uploaded_data = combined_data  # Use combined data
            st.session_state.auto_insights = generate_comprehensive_insights(combined_data)
            st.session_state.mock_data_generated = True
            st.success("Demo data loaded!")
            st.rerun()
        
        if st.button("🗑️ Clear Data"):
            st.session_state.uploaded_data = None
            st.session_state.auto_insights = []
            st.session_state.chat_history = []
            st.session_state.mock_data_generated = False
            st.rerun()
    
    # Main Content Area
    if tab == "📁 Data Upload":
        st.header("📁 Data Upload & Auto-Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Upload Your Business Data")
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Supported formats: CSV, Excel (.xlsx, .xls)"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("🔄 Processing your data..."):
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Clean and process data
                        df = df.dropna(how='all').reset_index(drop=True)
                        
                        issues = validate_data(df)
                        if issues:
                            st.warning("Data issues detected: " + "; ".join(issues))
                        
                        st.session_state.uploaded_data = df
                        st.session_state.auto_insights = generate_comprehensive_insights(df)
                        st.session_state.mock_data_generated = False
                    
                    st.success("🎉 Data uploaded and analyzed successfully!")
                    
                    # Show immediate insights
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Records", len(df))
                    with col_b:
                        st.metric("Columns", len(df.columns))
                    with col_c:
                        numeric_cols, _ = detect_column_types(df)
                        st.metric("Numeric Metrics", len(numeric_cols))
                    
                except ValueError as ve:
                    st.error(f"❌ Data validation error: {str(ve)}")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with col2:
            st.markdown("### 🚀 What You Get")
            features = [
                "✅ **Instant Data Analysis** - Automatic insights generation",
                "✅ **Smart Visualizations** - AI-powered chart recommendations", 
                "✅ **Natural Language Queries** - Ask questions in plain English",
                "✅ **Advanced Analytics** - Trend analysis, forecasting, anomaly detection",
                "✅ **Business Intelligence** - KPIs, dashboards, performance metrics"
            ]
            
            for feature in features:
                st.markdown(feature)
    
    elif tab == "📊 Overview Dashboard":
        st.header("📊 Business Intelligence Dashboard")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data or load sample data to view dashboard")
            return
        
        df = st.session_state.uploaded_data
        numeric_cols, categorical_cols = detect_column_types(df)
        date_cols = detect_date_columns(df)
        
        # KPI Cards Row
        st.subheader("🎯 Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            total_records = len(df)
            st.markdown(f"""
            <div class="overview-card">
                <h2 style="margin:0; font-size: 2rem;">{total_records:,}</h2>
                <p style="margin:0; opacity: 0.8;">Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col2:
            if numeric_cols:
                avg_value = df[numeric_cols[0]].mean()
                growth = 0
                if numeric_cols and date_cols:
                    df_sorted = df.sort_values(date_cols[0])
                    if len(df_sorted) > 1:
                        mid = len(df_sorted) // 2
                        recent_period = df_sorted.iloc[-mid:]
                        previous_period = df_sorted.iloc[:mid]
                        prev_mean = previous_period[numeric_cols[0]].mean()
                        if prev_mean != 0:
                            growth = ((recent_period[numeric_cols[0]].mean() - prev_mean) / prev_mean) * 100
                st.markdown(f"""
                <div class="overview-card">
                    <h2 style="margin:0; font-size: 2rem;">{avg_value:,.0f}</h2>
                    <p style="margin:0; opacity: 0.8;">Avg {numeric_cols[0]}</p>
                    <p style="margin:0; font-size: 0.8rem;">💰 {growth:+.1f}% vs previous period</p>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_col3:
            if categorical_cols:
                unique_categories = df[categorical_cols[0]].nunique()
                st.markdown(f"""
                <div class="overview-card">
                    <h2 style="margin:0; font-size: 2rem;">{unique_categories}</h2>
                    <p style="margin:0; opacity: 0.8;">Unique {categorical_cols[0]}</p>
                    <p style="margin:0; font-size: 0.8rem;">🎯 Diversified portfolio</p>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_col4:
            data_quality = (df.notna().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.markdown(f"""
            <div class="overview-card">
                <h2 style="margin:0; font-size: 2rem;">{data_quality:.1f}%</h2>
                <p style="margin:0; opacity: 0.8;">Data Quality</p>
                <p style="margin:0; font-size: 0.8rem;">✅ Excellent coverage</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto Insights
        st.subheader("🔑 Auto-Generated Insights")
        insights = st.session_state.auto_insights
        if insights:
            cols = st.columns(min(3, len(insights)))
            for i, insight in enumerate(insights):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h3 style="margin:0 0 0.5rem 0;">{insight['title']}</h3>
                        <p style="margin:0;">{insight['insight']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No insights generated yet. Try refreshing or uploading new data.")
        
        # Dashboard Visualization
        st.subheader("📈 Interactive Dashboard")
        fig = create_advanced_visualizations(df, 'overview_dashboard', None)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for dashboard visualization.")
        
        # Export Options
        st.subheader("📥 Export Data")
        if st.button("Download Processed Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")
        
        if st.session_state.auto_insights:
            if st.button("Download Insights as JSON"):
                json_data = json.dumps(st.session_state.auto_insights, indent=4)
                st.download_button("Download JSON", json_data, "insights.json", "application/json")
    
    elif tab == "🔍 Advanced Analytics":
        st.header("🔍 Advanced Analytics & Visualizations")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data or load sample data to perform advanced analytics")
            return
        
        df = st.session_state.uploaded_data
        numeric_cols, categorical_cols = detect_column_types(df)
        date_cols = detect_date_columns(df)
        
        # Interactive Filters
        st.subheader("🛠️ Analysis Filters")
        selected_numeric = st.selectbox("Select Numeric Column for Analysis", numeric_cols)
        selected_categorical = st.selectbox("Select Categorical Column", categorical_cols) if categorical_cols else None
        if date_cols:
            date_col = st.selectbox("Select Date Column", date_cols)
            min_date, max_date = df[date_col].min(), df[date_col].max()
            if isinstance(min_date, pd.Timestamp):
                min_date, max_date = min_date.date(), max_date.date()
            date_range = st.date_input("Select Date Range", [min_date, max_date])
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
        
        # Advanced Visualizations
        st.subheader("📊 Custom Visualizations")
        viz_type = st.selectbox("Select Visualization Type", ["Trend Analysis", "Scatter Plot", "Bar Chart", "Pie Chart"])
        
        fig = None
        if viz_type == "Trend Analysis" and selected_numeric:
            df_sorted = df.sort_index() if df.index.name else df.sort_values(date_cols[0]) if date_cols else df
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sorted.index if df.index.name else df_sorted[date_cols[0]] if date_cols else range(len(df)), y=df[selected_numeric], mode='lines+markers', name=selected_numeric))
            z = np.polyfit(range(len(df)), df[selected_numeric], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=df_sorted.index if df.index.name else df_sorted[date_cols[0]] if date_cols else range(len(df)), y=p(range(len(df))), mode='lines', name='Trend Line', line=dict(dash='dash')))
            fig.update_layout(title=f"{selected_numeric} Trend Analysis")
        
        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
            x_col = st.selectbox("X Axis", numeric_cols)
            y_col = st.selectbox("Y Axis", numeric_cols, index=1)
            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}")
        
        elif viz_type == "Bar Chart" and selected_categorical and selected_numeric:
            grouped = df.groupby(selected_categorical)[selected_numeric].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=grouped.index, y=grouped.values, title=f"Bar Chart: {selected_numeric} by {selected_categorical}")
        
        elif viz_type == "Pie Chart" and selected_categorical and selected_numeric:
            grouped = df.groupby(selected_categorical)[selected_numeric].sum().head(8)
            fig = px.pie(names=grouped.index, values=grouped.values, title=f"Pie Chart: {selected_numeric} Distribution by {selected_categorical}")
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select appropriate columns and visualization type to generate chart.")
        
        # Advanced Metrics
        st.subheader("📐 Advanced Metrics")
        if selected_numeric:
            metrics = calculate_advanced_metrics(df, [selected_numeric])[selected_numeric]
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total", f"{metrics['total']:,.2f}")
            with cols[1]:
                st.metric("Average", f"{metrics['average']:,.2f}")
            with cols[2]:
                st.metric("Growth Rate", f"{metrics['growth_rate']:.1f}%")
            with cols[3]:
                st.metric("Volatility", f"{metrics['volatility']:.1f}%")
    
    elif tab == "🤖 AI Assistant":
        st.header("🤖 AI-Powered Business Assistant")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data or load sample data to use the AI Assistant")
            return
        
        query = st.text_input("Ask a business question (e.g., 'What are my revenue trends?')", key="ai_query")
        if query:
            with st.spinner("Analyzing your query..."):
                response, chart = process_advanced_nlp_query(query, st.session_state.uploaded_data)
                st.session_state.chat_history.append({"query": query, "response": response})
                st.markdown(f"<div class='chat-user'>You: {query}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-ai'>AI: {response}</div>", unsafe_allow_html=True)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in reversed(st.session_state.chat_history[-10:]):  # Show last 10 interactions
                st.markdown(f"<div class='chat-user'>You: {chat['query']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-ai'>AI: {chat['response']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
