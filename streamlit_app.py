import streamlit as st
from BankETL import ETL_BofA_AI
from data_handlers import Transaction
from data_handlers import initialize_database
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from dotenv import main

main.load_dotenv()

def main():
    # Initialize the database
    initialize_database()
    transactions = Transaction()

    pg_bg_img = """
    <style>
    [data-testid="block-container"] {
    background-color: #171717;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #171717 12px ), repeating-linear-gradient( #12254555, #122545 );
    }
    
    </style>
    """
    sidebar_bg_img = """
    <style>
    [data-testid="stSidebarUserContent"] {
    background-color: #171717;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #171717 12px ), repeating-linear-gradient( #12254555, #122545 );
    }
    
    </style>
    """

    st.set_page_config(layout="wide",initial_sidebar_state="collapsed",page_title="Financial Dashboard")
    st.markdown(sidebar_bg_img, unsafe_allow_html=True)
    st.markdown(pg_bg_img, unsafe_allow_html=True)

    ### Sidebar ###

    st.sidebar.title("Menu")

    start_time = st.sidebar.date_input("Start Date", 
                                       value=(datetime.now()-timedelta(days=60)),
                                         min_value=pd.to_datetime('2023-08-01'), max_value=datetime.now()-timedelta(days=1)
                                         , key=None)
    end_time = st.sidebar.date_input("End Date",
                                      value=datetime.now(),
                                         min_value=pd.to_datetime('2023-08-02'), max_value=datetime.now()
                                         , key=None)
    
    categories = st.sidebar.multiselect("Categories", transactions.get_categories(),
                                         default=transactions.get_categories())

    account_name = st.sidebar.multiselect("Account", transactions.get_accounts(),default=transactions.get_accounts())


    # Initialize session state for tracking file upload
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    st.sidebar.markdown("""
                        ---
                        ## Upload new data
                        """)
    with st.sidebar.expander("Add more data here..."):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        # Check if a file is uploaded and it's different from the previously uploaded file
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            # Execute ETL process
            with st.spinner('Loading data...'):
                result = transactions.execute_etl_in_new_file(uploaded_file)
                print(result)
            if result.get('status') == 'error':
                st.error(result.get('message'))
            elif result.get('status') == 'success':
                st.success(result.get('message'))
            elif result.get('status') == 'warning':
                st.warning(result.get('message'))
            else:
                st.warning("Something went wrong... Unknown status was returned")
    

    ### Main Page ###

    st.markdown("""
    # :green[Financial Dashboard] :moneybag:
    """)

    # Expenses by category chart
    expenses_by_category = transactions.get_expenses_by_category(start_time, end_time,categories,account_name)
    fig1 = px.treemap(expenses_by_category, values='amount', path=['category_name'], title='Expenses by category')
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')

    # income vs expenses chart
    income_vs_expenses = transactions.get_income_vs_expenses(start_time, end_time, categories,account_name)
    fig2 = px.bar(income_vs_expenses, x='date', y=['income', 'expenses'], title='Income vs Expenses', barmode='group')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')

    # monthly expenses trend chart
    monthly_expenses_trend = transactions.get_monthly_expenses_trends(start_time, end_time, categories,account_name)
    fig3 = px.line(monthly_expenses_trend, x='date', y='amount', title='Monthly expenses trend', markers=True)
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')
    fig3.update_traces(marker=dict(size=12))

    # daily expenses trend chart
    daily_expenses_trend = transactions.get_daily_expenses_trends(start_time, end_time, categories,account_name)
    fig4 = px.line(daily_expenses_trend, x='date', y='amount', title='Daily expenses trend', markers=True)
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')
    fig4.update_traces(marker=dict(size=12))

    # top 10 expenses chart
    top_10_expenses = transactions.get_top_expenses(start_time, end_time, categories,account_name)
    fig5 = px.bar(top_10_expenses, y='short_description', x='amount', title='Top 10 expenses', orientation='h',hover_data=['description','amount'])
    fig5.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')
    

    # savings rate per month chart
    savings_rate = transactions.get_savings_rate_per_month(start_time, end_time, categories,account_name)
    fig6 = px.line(savings_rate, x='date', y='savings_rate', title='Savings rate per month', markers=True)
    fig6.update_layout(plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0.3)')

    # Add orange line at y=0
    fig6.add_shape(type="line", x0=savings_rate['date'].min(), y0=0, x1=savings_rate['date'].max(), y1=0,
                   line=dict(color="orange", width=2, dash="dash"))

    # Change color and size of points below 0
    fig6.update_traces(marker=dict(color=savings_rate['savings_rate'].apply(lambda x: 'orange' if x < 0 else 'lightblue'),
                                   size=12))  # Increase the size parameter to increase the radius of the circles

     # Create two columns
    col1, col2 = st.columns(2,gap="medium")

    # Display charts in columns
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig6, use_container_width=True)

    # Display transactions table
    transactions_list = transactions.load_transactions(start_time, end_time, categories, account_name)
    st.dataframe(transactions_list, use_container_width=True)

    ### AI assistant ###

    st.markdown("""
    # :green[AI Finance Assistant] 🤖
    """)          

    prompt = st.chat_input("Ask me anything about your finances data...")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner('thinking...'):
            placeholder_response = st.empty()
            assistant = ETL_BofA_AI()
            assistant.answer_question_from_db(prompt,placeholder_response)
            


if __name__ == "__main__":
    main()

