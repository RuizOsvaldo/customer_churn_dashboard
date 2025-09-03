import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and process data
@st.cache_data
def load_and_process_data():
    """Load and preprocess the customer churn data"""
    try:
        df = pd.read_csv("customer_churn.csv")
        if df.empty:
            st.error("Dataset is empty")
            return None, None, None, None, None
        
        # Your original preprocessing logic
        df = df.dropna()
        df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
        
        # Feature encoding
        df_encoded = pd.get_dummies(df.drop("customerID", axis=1), drop_first=True)
        X = df_encoded.drop("Churn", axis=1)
        y = df_encoded["Churn"]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return df, X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        st.error("customer_churn.csv file not found")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# Initialize models
@st.cache_resource
def get_models():
    """Initialize machine learning models"""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'SVM': SVC(random_state=42, probability=True)
    }
    return models

# Load data
data_result = load_and_process_data()
if data_result[0] is None:
    st.stop()

df, X_train, X_test, y_train, y_test = data_result

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose Analysis Section:",
    ["🏠 Overview", "📊 Data Exploration", "🤖 Model Comparison", "📈 Performance Analysis", "🔍 Feature Insights"]
)

# Main content based on page selection
if page == "🏠 Overview":
    st.title("📊 Enhanced Customer Churn Analysis")
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **Advanced churn prediction using multiple machine learning algorithms**
    
    ### 🔧 Technical Features:
    - **4 ML Algorithms**: Random Forest, Logistic Regression, Decision Tree, SVM
    - **Model Comparison**: Performance metrics and ROC curve analysis  
    - **Feature Analysis**: Importance rankings and correlation insights
    - **Interactive Visualizations**: Plotly-powered charts and insights
    
    ### 📈 Business Value:
    - **Predict Customer Churn**: Identify at-risk customers proactively
    - **Data-Driven Insights**: Understand key factors driving churn behavior
    - **Model Selection**: Compare algorithms to find optimal performance
    - **Strategic Planning**: Support customer retention initiatives
    """)
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        churn_rate = df["Churn"].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with col3:
        avg_tenure = df["tenure"].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    with col4:
        avg_charges = df["MonthlyCharges"].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
    
    # Sample data preview
    st.subheader("📄 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.info("👆 Use the sidebar to navigate through different analysis sections!")

elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Dataset Overview", "📈 Feature Distributions", "🎯 Churn Analysis"])
    
    with tab1:
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset statistics
            stats_df = pd.DataFrame({
                'Metric': ['Total Records', 'Features', 'Churned Customers', 'Retained Customers', 'Missing Values'],
                'Value': [
                    len(df),
                    len(df.columns) - 2,  # Exclude customerID and Churn
                    len(df[df['Churn'] == 1]),
                    len(df[df['Churn'] == 0]),
                    df.isnull().sum().sum()
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Churn distribution pie chart
            churn_counts = df['Churn'].value_counts()
            fig_pie = px.pie(
                values=churn_counts.values, 
                names=['Retained', 'Churned'],
                title="Customer Churn Distribution",
                color_discrete_map={'Retained': 'lightblue', 'Churned': 'salmon'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Data Insights")
        insights = [
            f"📊 Overall churn rate: {df['Churn'].mean():.1%}",
            f"📅 Average customer tenure: {df['tenure'].mean():.1f} months",
            f"💰 Average monthly charges: ${df['MonthlyCharges'].mean():.2f}",
            f"🔍 Data quality: {df.isnull().sum().sum()} missing values"
        ]
        for insight in insights:
            st.info(insight)
    
    with tab2:
        st.subheader("Feature Distributions")
        
        # Feature selection
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        selected_feature = st.selectbox("Select feature to analyze:", numeric_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram by churn status
            fig_hist = px.histogram(
                df, 
                x=selected_feature, 
                color='Churn',
                title=f"Distribution of {selected_feature}",
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={0: 'lightblue', 1: 'salmon'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by churn status
            fig_box = px.box(
                df, 
                x='Churn', 
                y=selected_feature,
                title=f"{selected_feature} by Churn Status",
                color='Churn',
                color_discrete_map={0: 'lightblue', 1: 'salmon'}
            )
            fig_box.update_xaxis(ticktext=['Retained', 'Churned'], tickvals=[0, 1])
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Feature statistics
        st.subheader(f"📊 {selected_feature} Statistics by Churn Status")
        churned_data = df[df['Churn'] == 1][selected_feature]
        retained_data = df[df['Churn'] == 0][selected_feature]
        
        stats_comparison = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Churned': [
                len(churned_data),
                churned_data.mean(),
                churned_data.median(),
                churned_data.std(),
                churned_data.min(),
                churned_data.max()
            ],
            'Retained': [
                len(retained_data),
                retained_data.mean(),
                retained_data.median(),
                retained_data.std(),
                retained_data.min(),
                retained_data.max()
            ]
        })
        st.dataframe(stats_comparison.round(2), use_container_width=True)
    
    with tab3:
        st.subheader("Churn Analysis by Categories")
        
        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService']
        
        cols = st.columns(2)
        
        for i, feature in enumerate(categorical_features):
            with cols[i % 2]:
                # Calculate churn rates by category
                churn_by_cat = df.groupby(feature)['Churn'].agg(['count', 'sum', 'mean']).reset_index()
                churn_by_cat.columns = [feature, 'Total', 'Churned', 'Churn_Rate']
                
                fig_bar = px.bar(
                    churn_by_cat, 
                    x=feature, 
                    y='Churn_Rate',
                    title=f"Churn Rate by {feature}",
                    color='Churn_Rate',
                    color_continuous_scale='reds',
                    text='Churn_Rate'
                )
                fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_bar.update_layout(yaxis_title="Churn Rate")
                fig_bar.update_yaxis(tickformat='.1%')
                st.plotly_chart(fig_bar, use_container_width=True)

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("---")
    
    st.subheader("🚀 Train Multiple ML Models")
    st.markdown("""
    Compare the performance of different machine learning algorithms:
    - **Random Forest**: Ensemble method with multiple decision trees
    - **Logistic Regression**: Linear classification with probability outputs
    - **Decision Tree**: Interpretable tree-based classification
    - **Support Vector Machine**: Advanced classification with kernel methods
    """)
    
    if st.button("🚀 Train All Models", type="primary"):
        models = get_models()
        
        with st.spinner("Training models... Please wait."):
            model_results = {}
            
            # Train and evaluate each model
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    model_results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'precision': report['weighted avg']['precision'],
                        'recall': report['weighted avg']['recall'],
                        'f1_score': report['weighted avg']['f1-score'],
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'classification_report': report,
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
            
            # Store results in session state
            st.session_state.model_results = model_results
            st.session_state.models_trained = True
        
        st.success("✅ All models trained successfully!")
    
    # Display results if available
    if st.session_state.get('models_trained', False):
        model_results = st.session_state.model_results
        
        st.markdown("---")
        
        # Performance metrics table
        st.subheader("📊 Model Performance Comparison")
        
        metrics_data = []
        for name, results in model_results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.3f}",
                'CV Score': f"{results['cv_mean']:.3f} (±{results['cv_std']:.3f})",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Model performance summary
        st.subheader("🏆 Performance Summary")
        best_accuracy = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy", best_accuracy[0], f"{best_accuracy[1]['accuracy']:.3f}")
        with col2:
            st.metric("Best F1-Score", best_f1[0], f"{best_f1[1]['f1_score']:.3f}")
        with col3:
            avg_accuracy = np.mean([r['accuracy'] for r in model_results.values()])
            st.metric("Average Accuracy", f"{avg_accuracy:.3f}")
        
        # ROC Curves
        st.subheader("📈 ROC Curve Comparison")
        
        fig_roc = go.Figure()
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (name, results) in enumerate(model_results.items()):
            if results['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{name} (AUC = {roc_auc:.3f})',
                    line=dict(color=colors[i % len(colors)])
                ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Performance comparison bar chart
        st.subheader("📊 Performance Metrics Visualization")
        
        # Prepare data for bar chart
        metrics_for_chart = []
        for name, results in model_results.items():
            metrics_for_chart.extend([
                {'Model': name, 'Metric': 'Accuracy', 'Score': results['accuracy']},
                {'Model': name, 'Metric': 'Precision', 'Score': results['precision']},
                {'Model': name, 'Metric': 'Recall', 'Score': results['recall']},
                {'Model': name, 'Metric': 'F1-Score', 'Score': results['f1_score']}
            ])
        
        chart_df = pd.DataFrame(metrics_for_chart)
        fig_comparison = px.bar(
            chart_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Metrics Comparison',
            height=500
        )
        fig_comparison.update_yaxis(range=[0, 1])
        st.plotly_chart(fig_comparison, use_container_width=True)

elif page == "📈 Performance Analysis":
    st.title("📈 Performance Analysis")
    st.markdown("---")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first in the Model Comparison section.")
        st.stop()
    
    model_results = st.session_state.model_results
    
    # Model selection for detailed analysis
    model_names = list(model_results.keys())
    selected_model = st.selectbox("Select model for detailed analysis:", model_names)
    
    if selected_model:
        results = model_results[selected_model]
        
        # Key metrics display
        st.subheader(f"📊 {selected_model} Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{results['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{results['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{results['f1_score']:.3f}")
        
        st.markdown("---")
        
        # Confusion matrix and classification report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Confusion Matrix")
            cm = results['confusion_matrix']
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title=f"Confusion Matrix - {selected_model}"
            )
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            fig_cm.update_xaxis(ticktext=['Retained', 'Churned'], tickvals=[0, 1])
            fig_cm.update_yaxis(ticktext=['Retained', 'Churned'], tickvals=[0, 1])
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Confusion matrix interpretation
            tn, fp, fn, tp = cm.ravel()
            st.markdown("**Matrix Values:**")
            st.markdown(f"- True Negatives (Correctly Retained): {tn}")
            st.markdown(f"- False Positives (Incorrectly Predicted Churn): {fp}")
            st.markdown(f"- False Negatives (Missed Churn): {fn}")
            st.markdown(f"- True Positives (Correctly Predicted Churn): {tp}")
        
        with col2:
            st.subheader("📋 Classification Report")
            
            # Format classification report for display
            class_report = results['classification_report']
            
            report_data = []
            for class_label, metrics in class_report.items():
                if isinstance(metrics, dict) and class_label not in ['accuracy']:
                    report_data.append({
                        'Class': 'Retained' if class_label == '0' else 'Churned' if class_label == '1' else class_label,
                        'Precision': f"{metrics['precision']:.3f}",
                        'Recall': f"{metrics['recall']:.3f}",
                        'F1-Score': f"{metrics['f1-score']:.3f}",
                        'Support': int(metrics['support']) if 'support' in metrics else ''
                    })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
            
            st.markdown("""
            **Metrics Explanation:**
            - **Precision**: Of predicted churns, how many were correct?
            - **Recall**: Of actual churns, how many were caught?
            - **F1-Score**: Harmonic mean of precision and recall
            - **Support**: Number of actual instances for each class
            """)
        
        st.markdown("---")
        
        # Cross-validation analysis
        st.subheader("🎯 Cross-Validation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CV statistics
            cv_mean = results['cv_mean']
            cv_std = results['cv_std']
            
            cv_stats = pd.DataFrame({
                'Metric': ['Mean Accuracy', 'Standard Deviation', 'Min Expected', 'Max Expected'],
                'Value': [
                    f"{cv_mean:.3f}",
                    f"{cv_std:.3f}",
                    f"{cv_mean - cv_std:.3f}",
                    f"{cv_mean + cv_std:.3f}"
                ]
            })
            st.dataframe(cv_stats, use_container_width=True)
            
            # Performance interpretation
            if cv_std < 0.05:
                st.success("✅ **Consistent Performance**: Low standard deviation indicates stable model")
            elif cv_std < 0.1:
                st.info("ℹ️ **Moderate Consistency**: Reasonable performance variation")
            else:
                st.warning("⚠️ **High Variation**: Model performance varies significantly across folds")
        
        with col2:
            # CV visualization
            cv_data = pd.DataFrame({
                'Metric': ['CV Mean', 'Test Accuracy'],
                'Score': [results['cv_mean'], results['accuracy']],
                'Type': ['Cross-Validation', 'Test Set']
            })
            
            fig_cv = px.bar(
                cv_data,
                x='Metric',
                y='Score',
                color='Type',
                title=f'Cross-Validation vs Test Performance - {selected_model}',
                height=400
            )
            fig_cv.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_cv, use_container_width=True)
        
        st.markdown("---")
        
        # Business impact analysis
        st.subheader("💼 Business Impact Analysis")
        
        # Calculate business metrics
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        total_customers = tn + fp + fn + tp
        actual_churn_rate = (tp + fn) / total_customers
        predicted_churn_rate = (tp + fp) / total_customers
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}")
        with col2:
            st.metric("Predicted Churn Rate", f"{predicted_churn_rate:.1%}")
        with col3:
            precision_churn = tp / (tp + fp) if (tp + fp) > 0 else 0
            st.metric("Churn Precision", f"{precision_churn:.1%}")
        with col4:
            recall_churn = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.metric("Churn Recall", f"{recall_churn:.1%}")
        
        # Business recommendations
        st.subheader("📝 Business Recommendations")
        
        recommendations = []
        
        if results['accuracy'] >= 0.85:
            recommendations.append("✅ **Strong Performance**: Model ready for production deployment")
        elif results['accuracy'] >= 0.75:
            recommendations.append("ℹ️ **Good Performance**: Consider feature engineering to improve further")
        else:
            recommendations.append("⚠️ **Needs Improvement**: Additional data or different algorithms recommended")
        
        if fn > tp * 0.3:  # High false negatives
            recommendations.append(f"⚠️ **Missed Opportunities**: {fn} churning customers not identified - consider lowering prediction threshold")
        
        if fp > tp * 0.5:  # High false positives
            recommendations.append(f"💰 **Cost Consideration**: {fp} customers incorrectly flagged - may increase retention costs")
        
        if recall_churn < 0.7:
            recommendations.append("🎯 **Focus on Recall**: Prioritize catching more churning customers to reduce revenue loss")
        
        for rec in recommendations:
            st.info(rec)

elif page == "🔍 Feature Insights":
    st.title("🔍 Feature Importance Analysis")
    st.markdown("---")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first in the Model Comparison section.")
        st.stop()
    
    model_results = st.session_state.model_results
    
    # Feature importance for tree-based models
    st.subheader("🌳 Feature Importance (Tree-based Models)")
    
    tree_models = ['Random Forest', 'Decision Tree']
    available_tree_models = [model for model in tree_models if model in model_results]
    
    if available_tree_models:
        selected_tree_model = st.selectbox("Select tree-based model:", available_tree_models)
        
        if selected_tree_model:
            model = model_results[selected_tree_model]['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                feature_importance = pd.Series(
                    model.feature_importances_, 
                    index=X_train.columns
                ).sort_values(ascending=False)
                
                # Clean feature names for display
                clean_names = {
                    "MonthlyCharges": "Monthly Charges",
                    "TotalCharges": "Total Charges", 
                    "SeniorCitizen": "Senior Citizen",
                    "Partner_Yes": "Has Partner",
                    "Dependents_Yes": "Has Dependents",
                    "PhoneService_Yes": "Phone Service",
                    "gender_Male": "Male Gender"
                }
                
                display_names = [clean_names.get(name, name.replace("_", " ").title()) for name in feature_importance.index]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Top 10 features
                    top_features = feature_importance.head(10)
                    top_display_names = display_names[:10]
                    
                    fig_importance = px.bar(
                        x=top_features.values,
                        y=top_display_names,
                        orientation='h',
                        title=f'Top 10 Feature Importance - {selected_tree_model}',
                        color=top_features.values,
                        color_continuous_scale='viridis'
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title='Importance Score',
                        yaxis_title='Features',
                        height=500
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    st.subheader("Top 5 Features")
                    for i, (importance, display_name) in enumerate(zip(top_features.head(5).values, top_display_names[:5]), 1):
                        st.metric(f"#{i} {display_name}", f"{importance:.4f}")
                
                # Complete feature importance table
                st.subheader("📋 Complete Feature Importance Rankings")
                
                importance_df = pd.DataFrame({
                    'Rank': range(1, len(feature_importance) + 1),
                    'Feature': display_names,
                    'Importance': feature_importance.values
                }).round(4)
                
                st.dataframe(importance_df, use_container_width=True)
    
    else:
        st.info("No tree-based models available for feature importance analysis")
    
    st.markdown("---")
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlation Analysis")
    
    # Get numeric features for correlation
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']
    
    if all(col in df.columns for col in numeric_cols):
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation with churn
        st.subheader("🎯 Correlation with Churn")
        churn_correlations = corr_matrix['Churn'].drop('Churn').abs().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Positive correlations (risk factors)
            positive_corr = corr_matrix['Churn'].drop('Churn')[corr_matrix['Churn'].drop('Churn') > 0].sort_values(ascending=False)
            if len(positive_corr) > 0:
                st.markdown("**Positive Correlations (↑ Churn Risk)**")
                pos_df = pd.DataFrame({
                    'Feature': ['Monthly Charges' if x == 'MonthlyCharges' else 'Senior Citizen' if x == 'SeniorCitizen' else x for x in positive_corr.index],
                    'Correlation': positive_corr.values
                }).round(3)
                st.dataframe(pos_df, use_container_width=True)
        
        with col2:
            # Negative correlations (protective factors)
            negative_corr = corr_matrix['Churn'].drop('Churn')[corr_matrix['Churn'].drop('Churn') < 0].sort_values()
            if len(negative_corr) > 0:
                st.markdown("**Negative Correlations (↓ Churn Risk)**")
                neg_df = pd.DataFrame({
                    'Feature': ['Total Charges' if x == 'TotalCharges' else x for x in negative_corr.index],
                    'Correlation': negative_corr.values
                }).round(3)
                st.dataframe(neg_df, use_container_width=True)
    
    else:
        st.info("Some features not available for correlation analysis")
    
    st.markdown("---")
    
    # Business insights and recommendations
    st.subheader("💡 Business Insights & Recommendations")
    
    # Generate insights based on available analysis
    st.markdown("### 🎯 Key Churn Drivers")
    
    insights = [
        "**Tenure**: Longer-tenured customers show significantly lower churn rates - focus on early customer engagement",
        "**Monthly Charges**: Higher monthly charges correlate with increased churn - review pricing strategy and value proposition",
        "**Customer Demographics**: Senior citizens may have different churn patterns - develop targeted retention programs",
        "**Service Adoption**: Phone service and other features impact churn likelihood - promote value-added services"
    ]
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")
    
    st.markdown("### 🚀 Strategic Recommendations")
    
    recommendations = [
        "**Proactive Identification**: Use model predictions to identify at-risk customers before they churn",
        "**Segmented Retention**: Develop different retention strategies based on customer profiles and risk factors",
        "**Early Warning System**: Monitor changes in key predictive features for real-time churn alerts",
        "**Value Communication**: Better communicate service value to customers with high monthly charges",
        "**Onboarding Enhancement**: Improve early customer experience to build long-term loyalty",
        "**A/B Testing**: Test different retention offers on predicted high-risk customer segments"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Enhanced Churn Analysis**")
st.sidebar.markdown("Multiple ML algorithms for comprehensive insights")

# Footer
st.markdown("---")
st.markdown("*Enhanced Customer Churn Analysis Dashboard - Powered by Python, scikit-learn & Streamlit*")
