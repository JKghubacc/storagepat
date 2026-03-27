import mariadb
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import os

os.makedirs("plots/flows", exist_ok=True)
os.makedirs("plots/stocks", exist_ok=True)
os.makedirs("plots/growth", exist_ok=True)
os.makedirs("plots/networks/full", exist_ok=True)
os.makedirs("plots/networks/top10", exist_ok=True)
os.makedirs("plots/forecast/network_fit", exist_ok=True)
os.makedirs("plots/forecast/arima_fit", exist_ok=True)
os.makedirs("plots/forecast/comparison", exist_ok=True)
os.makedirs("plots/forecast/comparison_panel", exist_ok=True)
os.makedirs("plots/forecast/extended", exist_ok=True)
os.makedirs("plots/forecast/extended_panel", exist_ok=True)
os.makedirs("plots/marginal_effects", exist_ok=True)
os.makedirs("plots/analysis", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# Dictionary with technology names and corresponding file paths
technologies = {
    "Pumpedhydro": "Pumpedhydro_data.csv",
    "Flywheel": "Flywheel_data.csv",
    "Liquidair": "Liquidair_data.csv",
    "Compressedair": "Compressedair_data.csv",
    "Drygravity": "Drygravity_data.csv",
    "Pumpedthermal": "Pumpedthermal_data.csv",
    "Redoxflow": "Redoxflow_data.csv",
    "Supercapacitors":"Supercapacitors_data.csv",
    "LithiumNickelCobaltOxide":"LithiumNickelCobaltOxide_data.csv",
    "InorganicSolidState":"InorganicSolidState_data.csv",
    "NickelManganeseCobalt": "Nickelmanganesecobalt_data.csv",
    "SodiumIon":"SodiumIon_data.csv",
    "SolarThermal":"SolarThermal.csv",
    "ThermalLiquid":"ThermalLiquid_data.csv",
    "ThermalLatent":"ThermalLatent_data.csv",
    "ThermalSolid":"ThermalSolid_data.csv",
    "SteamAccumulators":"SteamHeatAccumulators_data.csv"
}

cpc_depth = 4
offset_stock=10
max_priority_year = 2024

# Set to 1 for using logs, 0 not using 
logs=0

growth_array_corr=[]
neighborgrowth_array_corr=[]

growth_array_estim=[]
technology_ols_results=[]

estimation_year_start=1980
estimation_year_end=2023

margineffect_year= estimation_year_end
margincitframe=pd.DataFrame()

# Define the period (x years) for growth rate calculation (e.g., 1 year)
x = 1

# Create dataframe for results
results_df = pd.DataFrame(columns=['Technology', 'Intercept', 'Beta', 'R_squared', 'P_value_Beta', 'Std_Error_Beta'])

# Create frame for aggregated data
technology_data = {}

# Iterate over each technology and file path
for technology, file_path in technologies.items():
    
    # Read the CSV file into a DataFrame
    family_data = pd.read_csv(file_path)
    print(f'Number of patents for technology {technology}' + str(len(family_data)))
    # Set starting year to earliest filing year in case it is more recent than 1980

    original_estimation_year_start = estimation_year_start
    if family_data['Früheste Priorität'].min() > estimation_year_start:
        estimation_year_start = (family_data['Früheste Priorität'].min()) + 1

    original_estimation_year_end = estimation_year_end
    if family_data['Früheste Priorität'].max() < estimation_year_end:
        estimation_year_end = (family_data['Früheste Priorität'].max())

    # Generate stocks and flows of technologies patents
    flows = family_data.groupby('Früheste Priorität').size().reset_index(name='count')
    
    # Create a complete range of years (including any missing years)
    all_years = pd.DataFrame({'Früheste Priorität': range(flows['Früheste Priorität'].min(), flows['Früheste Priorität'].max() + 1)})
    
    # Merge with flows to include all years and fill missing values with 0
    all_years = all_years.merge(flows, on='Früheste Priorität', how='left').fillna(0)
    
    # Replace stocks for years with 0 flows by carrying forward the previous year's stock
    all_years['count'] = all_years['count'].fillna(0)
    
    # Now, calculate the cumulative stock after filling missing years
    all_years['Stock'] = all_years['count'].cumsum()

    # Include offset of a minimum stock
    all_years = all_years[all_years['Stock'] >= offset_stock]

    if all_years['Früheste Priorität'].min() > estimation_year_start:
        estimation_year_start = (all_years['Früheste Priorität'].min()) + 1

    # Calculate x-year growth rate of stocks
    if logs==1:
        all_years[f'Technology_Growth_Rate_{x}_Years'] = all_years['Stock'].pct_change(periods=x).apply(lambda pct: np.log(pct+1))
    else:
        all_years[f'Technology_Growth_Rate_{x}_Years'] = all_years['Stock'].pct_change(periods=x)
    print(all_years[f'Technology_Growth_Rate_{x}_Years'])
    # Plotting Flows
    plt.figure(figsize=(10, 6))
    plt.plot(all_years['Früheste Priorität'], all_years['count'], marker='o', color='blue', label='Flows')
    plt.title(f'Flows of patent families of technology {technology}')
    plt.xlabel('Year')
    plt.xticks(ticks=range(int(all_years['Früheste Priorität'].min()), int(all_years['Früheste Priorität'].max()) + 1,10))
    plt.ylabel('Number of patent families (Flows)')
    plt.grid()
    plt.legend()
    flows_plot_file = f'plots/flows/flows_{technology}.png'
    plt.savefig(flows_plot_file, dpi=300)
    plt.close()

    # Plotting Stocks
    plt.figure(figsize=(10, 6))
    plt.plot(all_years['Früheste Priorität'], all_years['Stock'], marker='o', color='blue', label='Stocks')
    plt.title(f'Stocks of patent families of technology {technology}')
    plt.xlabel('Year')
    plt.xticks(ticks=range(int(all_years['Früheste Priorität'].min()), int(all_years['Früheste Priorität'].max()) + 1,10))
    plt.ylabel('Cumulative Number of patent families')
    plt.grid()
    plt.legend()
    stocks_plot_file = f'plots/stocks/stocks_{technology}.png'
    plt.savefig(stocks_plot_file, dpi=300)
    plt.close()

    print(f"Plots saved as {flows_plot_file} and {stocks_plot_file}.")

    # Connect to the MariaDB database
    connection = mariadb.connect(
        user='',       # Your username
        host='localhost',
        database='patstat'
    )

    cursor = connection.cursor(dictionary=True)

    # List to store the results for family numbers
    all_results = []

    # Iterate over each family number in the data
    for family_number in family_data['Familiennummer']:
        try:
            # Retrieve the priority year for the current family number from the CSV file
            priority_year = family_data.loc[family_data['Familiennummer'] == family_number, 'Früheste Priorität'].values[0]

            # SQL query for each family number
            query = """
            SELECT 
                c.docdb_family_id,
                c.cited_docdb_family_id,
                d.cpc_class_symbol
            FROM tls228_docdb_fam_citn AS c
            LEFT JOIN tls225_docdb_fam_cpc AS d ON c.cited_docdb_family_id = d.docdb_family_id
            WHERE c.docdb_family_id = %s;
            """
            
            # Execute the query
            cursor.execute(query, (family_number,))
            
            # Fetch results in chunks
            while True:
                rows = cursor.fetchmany(1000)  # Fetch in batches
                if not rows:
                    break

                # Add the priority year to each record
                for row in rows:
                    row['Prioritätsjahr'] = priority_year

                all_results.extend(rows)
            
        except mariadb.Error as e:
            print(f"Error processing family number {family_number}: {e}")

    # Convert the results to a DataFrame
    citation_results = pd.DataFrame(all_results)

    # Drop Y codes from Citations
    citation_results = citation_results[~citation_results['cpc_class_symbol'].str.startswith('Y', na=False)]
    print(f'Number of citations for technology {technology}' + str(len(citation_results)))
    """Generating citation network"""

    # Count the number of CPC classes per cited family
    cited_class_counts = citation_results.groupby('cited_docdb_family_id')['cpc_class_symbol'].nunique().reset_index()
    cited_class_counts.columns = ['cited_docdb_family_id', 'class_count']

    # Merge the class count back into citation_results to normalize the citations
    citation_results = citation_results.merge(cited_class_counts, on='cited_docdb_family_id', how='left')

    # Normalize each citation by dividing by the number of classes for that cited family
    citation_results['normalized_citation'] = 1 / citation_results['class_count']

    # Truncate CPC codes to the desired depth (cpc_depth)
    citation_results['cpc_class_symbol_truncated'] = citation_results['cpc_class_symbol'].str[:cpc_depth]

    # Aggregate the normalized citations by the truncated CPC class symbols
    aggregated_citations = citation_results.groupby('cpc_class_symbol_truncated')['normalized_citation'].sum().reset_index()
    aggregated_citations.columns = ['cpc_class', 'total_normalized_citations']

    """Generate citation vector and process citation data"""
    def create_citation_vector(aggregated_citations):
        citation_vector = aggregated_citations.set_index('cpc_class')['total_normalized_citations'].sort_index()
        return citation_vector

    # Generate the citation vector
    citation_vector = create_citation_vector(aggregated_citations)
    total_sum = citation_vector.sum()
    
    # Normalize the citation vector
    citation_vector_normalized = citation_vector / total_sum
    #print(citation_vector_normalized)
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add central node for the technology
    G.add_node(technology, type='central')

    # Add nodes and edges for CPC classes with weights
    for code, count in citation_vector.items():
        G.add_node(code, type='code')  # Add CPC class as a node
        
        normalized_weight = count / total_sum  # Normalize the weight
        G.add_edge(technology, code, weight=normalized_weight)  # Add edge from technology to CPC class

    # Visualize the network
    pos = nx.spring_layout(G, seed=42)  # Position the nodes for layout
    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 6 for w in weights], edge_color='gray', arrowstyle='-|>')

    # Final plot settings
    plt.axis('off')
    plt.title('Citation Network for ' + technology)
    output_file = f'plots/networks/full/normalized_citation_network_{technology}_{max_priority_year}.png'
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

    """Visualize reduced network with only top 10 technologies"""
    print(citation_vector_normalized)
    top_10_citvector=citation_vector_normalized.sort_values(ascending=False)
    top_10_citvector=top_10_citvector[0:10]
    print(top_10_citvector)
    print(top_10_citvector.items())
    H = nx.DiGraph()
    H.add_node(technology, type='central')

    for code_H, count_H in top_10_citvector.items():
        H.add_node(code_H, type='code')  # Add CPC class as a node
        H.add_edge(technology, code_H, weight=count_H)

    # Visualize the new network
    pos_H = nx.spring_layout(H, seed=42)  # Position the nodes for layout
    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(
        H, pos_H, nodelist=[technology], node_color='lightblue', node_size=6000
    )
    other_nodes = [node for node in H.nodes if node != technology] 
    nx.draw_networkx_nodes(
        H, pos_H, nodelist=other_nodes, node_color='lightblue', node_size=1200
    )
    # Draw labels
    nx.draw_networkx_labels(H, pos_H, font_size=10, font_family='sans-serif')

    # Draw edges with weights
    edges_H = H.edges()
    print(edges_H)
    weights_H = [H[a][b]['weight'] for a, b in edges_H]
    print(weights_H)
    nx.draw_networkx_edges(H, pos_H, edgelist=edges_H, width=[c * 6 for c in weights_H], edge_color='gray', arrows=True, arrowstyle='-|>')

    # Final plot settings for the new network
    plt.text(0.5, -0.1, 'Edge thickness represents the normalized citation frequency', 
         horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)

    plt.axis('off')
    plt.title('Top 10 Cited Technologies Network for ' + technology)
    output_file_h = f'plots/networks/top10/top_10_cited_network_{technology}_{max_priority_year}.png'
    plt.savefig(output_file_h, format='png', dpi=300)
    plt.close()

    """Processing cited growth rates"""

    df = pd.read_csv('class_year_normalized_counts_4_digits.csv')

    # Remove entries where the year is 9999
    df = df[df['Year'] != 9999]

    # Sort by 'Class' and 'Year'
    df = df.sort_values(by=['Class', 'Year'])

    # Determine the range of years (min and max year)
    min_year = df['Year'].min()
    max_year = df['Year'].max()

    # Ensure all years are present for each class
    yearframe = pd.DataFrame({'Year': range(min_year, max_year + 1)})

    # Create a new DataFrame to store the filled data
    filled_df = pd.DataFrame()

    for cls in df['Class'].unique():
        # Filter the data for the current class
        class_df = df[df['Class'] == cls]
        
        # Fill missing years by merging with the full range of years
        class_df = pd.merge(yearframe, class_df, on='Year', how='left')
        
        # Assign the current class name to the 'Class' column
        class_df['Class'] = cls
        
        # Fill missing values in the 'Normalized_Count' column with 0
        class_df['Normalized_Count'] = class_df['Normalized_Count'].fillna(0)
        
        # Calculate the cumulative sum for this class
        class_df['Cumulative_Sum'] = class_df['Normalized_Count'].cumsum()
        
        # Add the results to the new DataFrame
        filled_df = pd.concat([filled_df, class_df])

    # Sort the new DataFrame by 'Class' and 'Year' again
    filled_df = filled_df.sort_values(by=['Class', 'Year'])

    # Calculate the growth rate over x years
    # With log
    if logs==1:
        filled_df[f'Growth_Rate_{x}_Years'] = filled_df.groupby('Class')['Cumulative_Sum'].pct_change(periods=x).apply(lambda pct: np.log(pct+1))
    # Without log
    else: 
        filled_df[f'Growth_Rate_{x}_Years'] = filled_df.groupby('Class')['Cumulative_Sum'].pct_change(periods=x)

    total_cumulative_sum = (
        filled_df.groupby('Year')['Cumulative_Sum']
        .sum()
        .reset_index()
        .rename(columns={'Cumulative_Sum': 'Total_Cumulative_Sum'})
    )

    filled_df = filled_df.merge(total_cumulative_sum, on='Year', how='left')

    total_growth_rate = (
        total_cumulative_sum['Total_Cumulative_Sum']
        .pct_change(periods=x)
        .rename(f'Total_Growth_Rate_{x}_Years')
    )

    total_cumulative_sum[f'Total_Growth_Rate_{x}_Years'] = total_growth_rate
    filled_df = filled_df.merge(total_cumulative_sum[['Year', f'Total_Growth_Rate_{x}_Years']], on='Year', how='left')

    filled_df[f'Normalized_Growth_Rate_{x}_Years'] = (
        filled_df[f'Growth_Rate_{x}_Years'] / filled_df[f'Total_Growth_Rate_{x}_Years']
    )
    # Remove rows with NaN values (these occur for the first years when calculating growth rate)
    filled_df = filled_df.dropna()

    """Analysis"""

    # Define the start and end year for growth calculation for ANNG
    start_year = 2013
    end_year = 2023

    original_start_year=start_year
    if all_years['Früheste Priorität'].min() > start_year:
        start_year = (all_years['Früheste Priorität'].min()) + 1

    # Step 1: Filter the data for the start and end year for each class
    start_year_data_classes = filled_df[filled_df['Year'] == start_year][['Class', 'Cumulative_Sum']]
    start_year_data_classes.rename(columns={'Cumulative_Sum': 'Stock_Start_Year'}, inplace=True)

    end_year_data_classes = filled_df[filled_df['Year'] == end_year][['Class', 'Cumulative_Sum']]
    end_year_data_classes.rename(columns={'Cumulative_Sum': 'Stock_End_Year'}, inplace=True)

    # Step 2: Merge the start and end year data to calculate growth
    growth_data_classes = pd.merge(start_year_data_classes, end_year_data_classes, on='Class', how='inner')

    # Calculate the growth rate for each class
    if logs==1:
        growth_data_classes['Growth_Rate'] = np.log(growth_data_classes['Stock_End_Year'] / growth_data_classes['Stock_Start_Year'])
        
    else:    
        growth_data_classes['Growth_Rate'] = (growth_data_classes['Stock_End_Year'] - growth_data_classes['Stock_Start_Year']) / growth_data_classes['Stock_Start_Year']

    # Step 3: Merge the citation vector with the growth data
    merged_data = pd.merge(citation_vector_normalized, growth_data_classes, left_index=True, right_on='Class', how='inner')

    # Step 4: Multiply the citation vector with the growth rate for each class
    merged_data['Weighted_Growth'] = merged_data['total_normalized_citations'] * merged_data['Growth_Rate']

    # Step 5: Calculate the overall weighted growth rate for the technology
    overall_weighted_growth = merged_data['Weighted_Growth'].sum()

    print(f"The overall weighted growth rate for {technology} between {start_year} and {end_year} is: {overall_weighted_growth}")

    # Get growth rate of the technology
    start_year_stock_technology = all_years[all_years['Früheste Priorität'] == start_year]['Stock'].values[0]
    end_year_stock_technology = all_years[all_years['Früheste Priorität'] == end_year]['Stock'].values[0]

    # Calculate growth
    growth_technology = np.log((end_year_stock_technology) / start_year_stock_technology)

    print(f'The overall growth rate for {technology} is: {growth_technology}')

    print(f"Processed data for {technology} completed.")
    growth_array_corr.append(growth_technology)
    neighborgrowth_array_corr.append(overall_weighted_growth)
    
    """Set up time series analysis for technology"""
    
    # Set up arrays for regression
    g_t_array = []
    x_t_array = []
    
    for t in range(estimation_year_start,estimation_year_end+1):
        g_t = all_years.loc[all_years['Früheste Priorität'] == t, f'Technology_Growth_Rate_{x}_Years'].values[0]
        g_t_array.append(g_t)
        citation_results_time=citation_results[citation_results['Prioritätsjahr']<=t]
        
        # Aggregate citations until time
        aggregated_citations_time = citation_results_time.groupby('cpc_class_symbol_truncated')['normalized_citation'].sum().reset_index()
        aggregated_citations_time.columns = ['cpc_class', 'total_normalized_citations']
        # Generate the citation vector until time t
        citation_vector_time = create_citation_vector(aggregated_citations_time)
        total_sum_time = citation_vector_time.sum()
        # Normalize the citation vector until t
        citation_vector_normalized_time = citation_vector_time / total_sum_time
        #print(citation_vector_normalized_time)
        growth_data_classes_time=filled_df[filled_df['Year']==t-1]
        # Step 3: Merge the citation vector with the growth data
        w_g_frame= pd.merge(citation_vector_normalized_time, growth_data_classes_time, left_index=True, right_on='Class', how='inner')

        # Step 4: Multiply the citation vector with the growth rate for each class
        w_g_frame['Weighted_Growth'] = w_g_frame['total_normalized_citations'] * w_g_frame[f'Growth_Rate_{x}_Years']
        
        anng_time=w_g_frame['Weighted_Growth'].sum()
        x_t_array.append(anng_time)
    
    #print('g_t_array is of length'+str(len(g_t_array)))
    #print('x_t_array is of length'+str(len(x_t_array)))
    #print('stock data is of length'+str(len(estimationdata_technology['Stock'])))

    temp_data = pd.DataFrame({
    'Year': list(range(estimation_year_start, estimation_year_end+1)),
    'Technology_Growth': g_t_array,
    'Weighted_Cited_Classes_Growth': x_t_array,
    'Stock':all_years[(all_years['Früheste Priorität'] <= estimation_year_end) & 
        (all_years['Früheste Priorität'] >= estimation_year_start)]['Stock']})
    
    #print(f"Growth rates for {technology}:")
    #print(temp_data[['Year', 'Technology_Growth']])
    plt.figure(figsize=(10, 6))
    plt.plot(temp_data['Year'], temp_data['Technology_Growth'], marker='o', color='blue', label='Growth data')
    plt.title(f'Relative growth of the stock patent families of technology {technology} over time')
    plt.xlabel('Year')
    plt.ylabel('Growth')
    plt.grid()
    plt.legend()
    # Save the plot
    growth_plot_file = f'plots/growth/Growth_{technology}.png'
    plt.savefig(growth_plot_file, dpi=300)
    plt.close()  # Close the plot to avoid display
    """Regression for all available data"""
    # Define the dependent variable (Technology Growth) and the independent variable (Cited Classes Growth)
    X = sm.add_constant(temp_data['Weighted_Cited_Classes_Growth'])  # Add constant for the intercept
    y = temp_data['Technology_Growth']
    
    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Output the summary for the current technology
    print(f"Regression results for technology: {technology}")
    #print(results.summary())
    technology_ols_results.append(results)
    # Get the OLS results (Intercept, Beta, R-squared, etc.)
    intercept = results.params['const']
    beta = results.params['Weighted_Cited_Classes_Growth']
    r_squared = results.rsquared
    p_value_beta = results.pvalues['Weighted_Cited_Classes_Growth']
    std_error_beta = results.bse['Weighted_Cited_Classes_Growth']
    
    # Append the results for the current technology to the DataFrame
    results_df = results_df.append({
        'Technology': technology,
        'Intercept': intercept,
        'Beta': beta,
        'R_squared': r_squared,
        'P_value_Beta': p_value_beta,
        'Std_Error_Beta': std_error_beta
    }, ignore_index=True)
    
    technology_data[technology] = temp_data

    #Reset estimation years 
    estimation_year_start = original_estimation_year_start
    estimation_year_end = original_estimation_year_end
    start_year=original_start_year

    citation_results_marginal=citation_results[citation_results['Prioritätsjahr']<=margineffect_year]
    # Aggregate citations until time
    aggregated_citations_marginal = citation_results_marginal.groupby('cpc_class_symbol_truncated')['normalized_citation'].sum().reset_index()
    aggregated_citations_marginal.columns = ['cpc_class', 'total_normalized_citations']
    # Generate the citation vector until time t
    citation_vector_marginal = create_citation_vector(aggregated_citations_marginal)
    total_sum_marginal = citation_vector_marginal.sum()
    # Normalize the citation vector until t
    citation_vector_normalized_marginal = citation_vector_marginal / total_sum_marginal
    print(citation_vector_normalized_marginal.head)
    print(citation_vector_normalized_marginal.index)
    print(citation_vector_normalized_marginal.values[0])
    margincitframe_temp=pd.DataFrame({'Technology':technology,
                                 'Cited Technology Class':citation_vector_normalized_marginal.index,
                                 'Normalized citations':citation_vector_normalized_marginal.values})
    margincitframe = pd.concat([margincitframe, margincitframe_temp], ignore_index=True)
    
cursor.close()
connection.close()

margincitframe.to_csv(f'Citationvectors_{margineffect_year}.csv')

"""ANNG analysis"""
neighborgrowth_array_corr = np.array(neighborgrowth_array_corr)
growth_array_corr= np.array(growth_array_corr)
g_anng_corr=np.corrcoef(growth_array_corr, neighborgrowth_array_corr)
print('Correlation between g and ANNG is '+ str(g_anng_corr))
# Scatterplot
plt.figure()
plt.scatter(neighborgrowth_array_corr, growth_array_corr, label='Data points')
# Regression
X_anng = sm.add_constant(neighborgrowth_array_corr) 
model_anng = sm.OLS(growth_array_corr, X_anng).fit()
slope = model_anng.params[1]
intercept = model_anng.params[0]
r_squared = model_anng.rsquared
p_value = model_anng.pvalues[1]
# Regression line
regression_line = slope * neighborgrowth_array_corr + intercept
# Plot regression of g-ANNG
plt.plot(neighborgrowth_array_corr, regression_line, color='green', label='Regression line')
# Axis
plt.ylabel('Technology-Growth between ' + str(start_year) + ' and ' + str(end_year))
plt.xlabel('Average nearest neighbor technology growth between ' + str(start_year) + ' and ' + str(end_year))
# Put equations into plot
equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}\nP-Value = {p_value:.3f}'
plt.text(0.5, 0.9, equation_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.savefig('Scatter_g_anng_with_regression.png', dpi=300)
#print(model.summary())

#print(results_df)

"""Process time series of different technologies"""
# Concatenate all the temporary DataFrames into one
all_technology_data = pd.concat(technology_data.values(), keys=technology_data.keys(), names=['Technology', 'Index']).reset_index()
# Calculate the total stock for each year
all_technology_data['Total_Stock'] = all_technology_data.groupby('Year')['Stock'].transform('sum')

"""Panel regression"""
# Prepare panel data: Set MultiIndex with 'Technology' (entity) and 'Year' (time)
all_technology_data.set_index(['Technology', 'Year'], inplace=True)
# Define the dependent variable (y) and the independent variable (X)
# Filtere die Daten bis einschließlich 2018
X_panel = all_technology_data['Weighted_Cited_Classes_Growth']
y_panel = all_technology_data['Technology_Growth']
# Add a constant to the independent variable for the intercept
X_panel = sm.add_constant(X_panel)
# Apply the Fixed Effects model by setting entity_effects=True
model_panel = PanelOLS(y_panel, X_panel, entity_effects=True, check_rank=True)
# Fit the model
results_panel = model_panel.fit()
# Display the results
print(results_panel)
constant_term_panel = results_panel.params['const']
entity_effects = results_panel.estimated_effects
entity_effects_df = entity_effects.groupby(level='Technology').first()
entity_effects_df['TechnologyEffect']=entity_effects_df['estimated_effects'] + constant_term_panel
print(entity_effects_df)
std_errors_panel = results_panel.std_errors
print(std_errors_panel)
beta_weighted_cited_classes_growth = results_panel.params['Weighted_Cited_Classes_Growth']
print(beta_weighted_cited_classes_growth)

# LaTeX-output with Stargazer
stargazer = Stargazer([results_panel])
print(stargazer.render_latex())
with open('tables/regression_results_stargazer.tex', 'w') as file:
    file.write(stargazer.render_latex())

# Show entity effects of panel regression and save them
#print(entity_effects)

"""Total storage patent regression"""
# Calculate the share of each technology in the total stock
all_technology_data['Technology_Share'] = all_technology_data['Stock'] / all_technology_data['Total_Stock']
if logs==1:
    all_technology_data['Total_Stock_Growth']=all_technology_data['Total_Stock'].pct_change(periods=x).apply(lambda pct: np.log(pct+1))
else:
    all_technology_data['Total_Stock_Growth']=all_technology_data['Total_Stock'].pct_change(periods=x)
all_technology_data=all_technology_data.reset_index()

"""Calculation of m_j,t"""
betas_df = results_df[['Technology', 'Beta']] 
fixedeffects_df = results_df[['Technology', 'Intercept']]
technology_share_df = all_technology_data[all_technology_data['Year'] == margineffect_year-1][['Technology', 'Technology_Share']]

#print(margincitframe.columns)
margineffect_frame = pd.merge(technology_share_df, betas_df, on='Technology', how='left')
margineffect_frame['Product_p_beta']=margineffect_frame['Beta']*margineffect_frame['Technology_Share']
#print(margineffect_frame.head)
margincitframe_process=pd.DataFrame()
marginresults=pd.DataFrame()
for margin_cpc_class in margincitframe['Cited Technology Class'].unique():
    margincitframe_process=margincitframe[margincitframe['Cited Technology Class']==margin_cpc_class]
    #print(margincitframe_temp)
    margineffect_temp=pd.merge(margineffect_frame, margincitframe_process, how="left", left_on='Technology', right_on='Technology')
    #print(margineffect_temp)
    margineffect_temp['Marginaleffect_IndividualBeta']=margineffect_temp['Product_p_beta'] * margineffect_temp['Normalized citations']
    margineffect_temp['Marginaleffect_GlobalBeta']= beta_weighted_cited_classes_growth * margineffect_temp['Technology_Share']* margineffect_temp['Normalized citations']
    #print(margineffect_temp)
    marginresults=marginresults.append(margineffect_temp, ignore_index=True)
#print(marginresults)

marginresults_listed_individual=marginresults.groupby('Cited Technology Class')['Marginaleffect_IndividualBeta'].sum().reset_index().sort_values(by='Marginaleffect_IndividualBeta', ascending=False)
marginresults_listed_global=marginresults.groupby('Cited Technology Class')['Marginaleffect_GlobalBeta'].sum().reset_index().sort_values(by='Marginaleffect_GlobalBeta', ascending=False)
print(marginresults_listed_individual.iloc[0:20])
print(marginresults_listed_global.iloc[0:20])

# Plot for 'Marginaleffect_IndividualBeta'
plt.figure(figsize=(10, 6))
plt.hist(marginresults_listed_individual['Marginaleffect_IndividualBeta'], bins=100, edgecolor = 'black')
plt.xlabel('Marginal Effect (Individual Beta)')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Histogram of Marginal Effects - Individual')
plt.savefig('marginaleffects_individual.png', dpi=300)
plt.close()

# Plot for 'Marginaleffect_GlobalBeta'
plt.figure(figsize=(10, 6))
plt.hist(marginresults_listed_global['Marginaleffect_GlobalBeta'], bins=100, edgecolor = 'black')
plt.xlabel('Marginal Effect (Global Beta)')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('Histogram of Marginal Effects - Global')
plt.savefig('marginaleffects_global.png', dpi=300)
plt.close()

"""Model comparison and forecasting"""
technologies_iter = all_technology_data['Technology'].unique()
original_cutoff_training=2018
forecast_end_year=2023
extended_forecast_end_year = 2030
rmse_network_individual_per_technology = []
rmse_ARIMA_per_technology = []
rmse_network_panel_per_technology = []
better_model_per_technology_individual = []
better_model_per_technology_panel = []
unconditional_network_forecast_2030_sums = []
unconditional_panel_network_forecast_2030_sums= []
results_2018_fit=[]
results_2018_parameters=pd.DataFrame(columns=['Technology', 'Intercept', 'Beta', 'R_squared', 'P_value_Beta', 'Std_Error_Beta'])

# Process growth data of 4-digit cpc technologies
# Process predicted growth of cited technology classes 
citedtechnologies_growthforecast = filled_df[['Class', 'Year', f'Growth_Rate_{x}_Years']].dropna()
citedtechnologies_growthforecast = citedtechnologies_growthforecast.replace([np.inf, -np.inf], np.nan).dropna()
# Filter data to include only years from 1980 onwards
citedtechnologies_growthforecast = citedtechnologies_growthforecast[citedtechnologies_growthforecast['Year'] >= 1980]
# Initialize an empty DataFrame to hold forecasted data
forecasted_df = pd.DataFrame()
# Loop over each unique class in the 'Class' column to generate forecasts
for forecast_cpcclass in citedtechnologies_growthforecast['Class'].unique():
    # Extract data for the current class
    cpc_df = citedtechnologies_growthforecast[citedtechnologies_growthforecast['Class'] == forecast_cpcclass]
    # Fit a linear regression model to the existing data
    X = cpc_df['Year'].values.reshape(-1, 1)
    y = cpc_df[f'Growth_Rate_{x}_Years'].values
    model = LinearRegression().fit(X, y)
    # Generate forecasted years and predict growth rates for each
    forecast_years = list(range(cpc_df['Year'].max() + 1, 2031))
    forecasted_growth_rates = model.predict([[year] for year in forecast_years])
    # Create a DataFrame for forecast years and growth rates
    forecast_df = pd.DataFrame({
        'Class': forecast_cpcclass,
        'Year': forecast_years,
        f'Growth_Rate_{x}_Years': forecasted_growth_rates
    })
    # Combine the original data with forecast data for the current class
    combined_df = pd.concat([cpc_df, forecast_df])
    forecasted_df = pd.concat([forecasted_df, combined_df])
# Reset index and save the final DataFrame to a CSV file
forecasted_df.reset_index(drop=True, inplace=True)
#forecasted_df.to_csv('Forecasts_4digit_cpc.csv')
print(forecasted_df)
all_technology_data.to_csv('alltechnology_data.csv')
arima_results_df=[]
"Network and ARIMA time series regression and comparison"
for tech in technologies_iter:
    # Filter data for the current technology (select by first index level)
    tech_data = all_technology_data[all_technology_data['Technology']==tech]
    cutoff_training=original_cutoff_training
    #if tech_data['Year'].max()<original_cutoff_training:
        #cutoff_training=tech_data['Year'].max()-5 
    tech_data_cut = tech_data[tech_data['Year'] <= cutoff_training]
    # Extract the 'Year' from the MultiIndex for the x-axis
    years=tech_data['Year']
    years_cut = tech_data_cut['Year']
    # Define X (independent variable) and y (dependent variable) for the current technology
    X_comparison = tech_data_cut['Weighted_Cited_Classes_Growth']
    y_comparison = tech_data_cut['Technology_Growth'].values
    # Add a constant to X to account for the intercept in the regression model
    X_comparison = sm.add_constant(X_comparison)
    #Perform linear regression
    model_comparison = sm.OLS(y_comparison, X_comparison)  # Create the OLS model
    results = model_comparison.fit()  # Fit the model to get the parameters

    """Save ols regression results of each technology"""
    stargazer_ols_technology = Stargazer([results])

    stargazer_ols_technology.title(f"Regression Results for storage technology {tech} (individual estimation)")

    with open(f'tables/technology_regression_results_stargazer_{tech}.tex', 'w') as file:
        file.write(stargazer_ols_technology.render_latex())
    # Get the OLS results (Intercept, Beta, R-squared, etc.)
    intercept = results.params['const']
    beta = results.params['Weighted_Cited_Classes_Growth']
    r_squared = results.rsquared
    p_value_beta = results.pvalues['Weighted_Cited_Classes_Growth']
    std_error_beta = results.bse['Weighted_Cited_Classes_Growth']
    
    # Append the results for the current technology to the DataFrame
    results_2018_parameters = results_2018_parameters.append({
        'Technology': tech,
        'Intercept': intercept,
        'Beta': beta,
        'R_squared': r_squared,
        'P_value_Beta': p_value_beta,
        'Std_Error_Beta': std_error_beta
    }, ignore_index=True)
    #Make predictions based on the model
    y_pred = results.predict(X_comparison).values  # Predicted values (y_hat)
    #Plot actual vs predicted values with own model
    plt.figure(figsize=(10,6))
    # Plot the actual values (scatter plot) with Year on the x-axis
    plt.plot(years_cut, y_comparison, color='blue', label='Actual Values', marker='o')
    # Plot the predicted values (line plot) with Year on the x-axis
    plt.plot(years_cut, y_pred, color='red', label='Predicted Values', linewidth=2)
    # Add plot details
    plt.title(f'Technology: {tech} - Time series vs. Network model estimation')
    plt.xlabel('Year')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(years_cut.min(),years_cut.max()+1,5))
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/forecast/network_fit/NetworkModelComparison_{tech}_{cutoff_training}.png', dpi=300)
    plt.close()

    # Set up test data for comparison
    y_test = tech_data[(tech_data['Year'] > cutoff_training) & (tech_data['Year'] <= forecast_end_year)]['Technology_Growth'].values

    "Network model forecast for test data"
    # Conditional forecast with empirical X_t
    X_forecast=tech_data[(tech_data['Year']>cutoff_training) & (tech_data['Year']<=forecast_end_year)]['Weighted_Cited_Classes_Growth']
    X_forecast = sm.add_constant(X_forecast)
    y_forecast_network = results.predict(X_forecast)

    # Unconditional forecast until 2030
    # Initialize citations of technology, assuming stable dependency
    citationvector_unconditional=margincitframe[margincitframe['Technology']==tech]
    unconditional_anng=pd.merge(citationvector_unconditional, forecasted_df, how='left', left_on='Cited Technology Class', right_on='Class')
    unconditional_anng['Relativegrowth']=unconditional_anng['Normalized citations'] * unconditional_anng[f'Growth_Rate_{x}_Years']
    anng_forecasted=[]
    for t_forecast in range(forecast_end_year+1, extended_forecast_end_year+1):
        temp_anng=unconditional_anng[unconditional_anng['Year']==t_forecast]['Relativegrowth'].sum()
        anng_forecasted.append(temp_anng)
    anng_forecasted=sm.add_constant(anng_forecasted)
    unconditional_forecasted_y=results.predict(anng_forecasted)

    unconditional_network_forecast_2030_sum = (np.product(1+unconditional_forecasted_y))-1
    unconditional_network_forecast_2030_sums.append((tech, unconditional_network_forecast_2030_sum))

    "ARIMA Regression"

    # Build and fit the ARIMA/ARIMA/AutoARIMA model

    model_fit = auto_arima(y_comparison, start_p=0, start_q=0, seasonal=False)

    print(f"ARIMA model summary for {tech}:\n", model_fit.summary())
    params = model_fit.params()
    stderr = model_fit.bse()
    pvalues = model_fit.pvalues()
    
    order = model_fit.order 
    seasonal_order = model_fit.seasonal_order 


    param_names = model_fit.arima_res_.param_names

    tech_results_df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': params,
        'Std Error': stderr,
        'p-value': pvalues,
        'Technology': tech,
        'ARIMA Order': [order] * len(params)
    })
    arima_results_df.append(tech_results_df)
    # Get the fitted values
    fitted_values = model_fit.predict_in_sample()
    # Plot the actual data and the fitted values with the ARIMA model
    plt.figure(figsize=(10, 6))
    plt.plot(years_cut, y_comparison, label='Actual Technology Growth', marker='o')
    plt.plot(years_cut, fitted_values, label='ARIMA Fitted Values', color='green', marker='o')
    plt.title(f'ARIMA Fitted Regression for {tech} Technology Growth')
    plt.xlabel('Year')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(years_cut.min(),years_cut.max()+1,5))
    plt.legend()
    plt.savefig(f'plots/forecast/arima_fit/ARIMA_{tech}_{cutoff_training}_comparison.png', dpi=300)
    plt.close()

    # Define future years for plotting

    future_years= np.arange(cutoff_training + 1, forecast_end_year + 1)

    start_prediction = cutoff_training + 1
    end_prediction = forecast_end_year
    extended_start_prediction = forecast_end_year + 1
    extended_end_prediction = extended_forecast_end_year

    forecast_steps = end_prediction - start_prediction + 1
    forecast_ARIMA = model_fit.predict(n_periods=forecast_steps)

    # Extract actual values and corresponding years for the entire period
    actual_values = tech_data['Technology_Growth']
    actual_years = tech_data['Year']

    # Plotting
    plt.figure(figsize=(10, 6))

    # Actual values (entire period)
    plt.plot(actual_years, actual_values, color='blue', label='Actual Values', marker='o')

    # Network model forecast (for test set period)
    plt.plot(future_years, y_forecast_network, color='red', label='Network Model Forecast', linewidth=2)

    # ARIMA forecast (for test set period)
    plt.plot(future_years, forecast_ARIMA, color='green', label='ARIMA Forecast', linewidth=2)

    # Add plot details
    plt.title(f'Technology: {tech} - Actual vs Forecasted Technology Growth')
    plt.xlabel('Year')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(actual_years.min(),future_years.max()+1,5))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/forecast/comparison/ForecastComparison_{tech}_{cutoff_training}.png', dpi=300)
    plt.close()

    # Forecast to 2030 for ARIMA

    extended_forecast_steps = extended_forecast_end_year - forecast_end_year
    forecast_ARIMA_extended = model_fit.predict(n_periods=extended_forecast_steps)

    future_years_extended= np.arange(forecast_end_year+1, extended_forecast_end_year + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(actual_years, actual_values, color='blue', label='Actual Technology Growth', marker='o')

    plt.plot(future_years_extended, forecast_ARIMA_extended, color='green', label='ARIMA Forecast', linewidth=2)

    plt.plot(future_years_extended, unconditional_forecasted_y, color='purple', linestyle='--', label='Unconditional Forecast until 2030', linewidth=2)

    plt.title(f'Technology: {tech} - Network model and ARIMA Forecast to 2030')
    plt.xlabel('Year')
    plt.axvline(x = 2023, color = 'lightskyblue')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(actual_years.min(),future_years_extended.max()+1,5))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/forecast/extended/Extended_NetworkAndARIMA_Forecast_{tech}_{cutoff_training}_to_2030.png', dpi=300)
    plt.close()

    # Conditional Forecast using PanelOLS parameters
    panel_forecast_df_conditional=pd.DataFrame({
            'Technology': tech,
            'Year': future_years,
            'Empirical_Cited_Classes_Growth': X_forecast['Weighted_Cited_Classes_Growth'].values
        }).set_index(['Technology', 'Year'])
    panel_forecast_df_conditional['Entity_Effect'] = entity_effects_df.loc[tech, 'estimated_effects']

    panel_forecast_df_conditional['Forecast_Technology_Growth'] = panel_forecast_df_conditional['Entity_Effect'] + (beta_weighted_cited_classes_growth * panel_forecast_df_conditional['Empirical_Cited_Classes_Growth']) + constant_term_panel

    # Unconditional forecast with PanelOLS parameters
    panel_forecast_df_unconditional = pd.DataFrame({
            'Technology': tech,
            'Year': future_years_extended,
            'Forecast_Cited_Classes_Growth': anng_forecasted[:, 1]
        }).set_index(['Technology', 'Year'])
        
    # Retrieve the fixed effect for this technology
    panel_forecast_df_unconditional['Entity_Effect'] = entity_effects_df.loc[tech, 'estimated_effects']
        
    # Calculate the forecast based on the global beta and the fixed effect
    panel_forecast_df_unconditional['Forecast_Technology_Growth'] = panel_forecast_df_unconditional['Entity_Effect'] + (beta_weighted_cited_classes_growth * panel_forecast_df_unconditional['Forecast_Cited_Classes_Growth']) + constant_term_panel
    
    
    unconditional_panel_network_forecast_2030_sum = (np.product(1+(panel_forecast_df_unconditional['Forecast_Technology_Growth'].values)))-1
    unconditional_panel_network_forecast_2030_sums.append((tech, unconditional_panel_network_forecast_2030_sum))


    # Plotting conditional forecast on the test set until 2023 with panel model estimation

    plt.figure(figsize=(10, 6))

    # Actual values (entire period)
    plt.plot(actual_years, actual_values, color='blue', label='Actual Values', marker='o')

    # Network model forecast (for test set period) with panel estimation
    plt.plot(future_years, panel_forecast_df_conditional['Forecast_Technology_Growth'].values, color='red', label='Network Model Forecast (panel estimation)', linewidth=2)

    # ARIMA forecast (for test set period)
    plt.plot(future_years, forecast_ARIMA, color='green', label='ARIMA Forecast', linewidth=2)

    # Add plot details
    plt.title(f'Technology: {tech} - Actual vs Forecasted Technology Growth')
    plt.xlabel('Year')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(actual_years.min(),future_years.max()+1,5))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/forecast/comparison_panel/ForecastComparison_Panel_{tech}_{cutoff_training}.png', dpi=300)
    plt.close()

    # Plotting unconditional panel forecast until 2030
    plt.figure(figsize=(10, 6))

    plt.plot(actual_years, actual_values, color='blue', label='Actual Technology Growth', marker='o')

    plt.plot(future_years_extended, forecast_ARIMA_extended, color='green', label='ARIMA Forecast', linewidth=2)

    plt.plot(future_years_extended, panel_forecast_df_unconditional['Forecast_Technology_Growth'], color='purple', linestyle='--', label='Unconditional Panel-Forecast until 2030', linewidth=2)

    plt.title(f'Technology: {tech} - Network model (Panel) and ARIMA Forecast to 2030')
    plt.xlabel('Year')
    plt.axvline(x = 2023, color = 'lightskyblue')
    plt.ylabel('Technology Growth')
    plt.xticks(ticks=range(actual_years.min(),future_years_extended.max()+1,5))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/forecast/extended_panel/PanelNetworkAndARIMA_Forecast_{tech}_{cutoff_training}_to_2030.png', dpi=300)
    plt.close()


    rmse_network_individual = np.sqrt(mean_squared_error(y_test, y_forecast_network))
    rmse_network_panel=np.sqrt(mean_squared_error(y_test,panel_forecast_df_conditional['Forecast_Technology_Growth'].values))
    rmse_ARIMA = np.sqrt(mean_squared_error(y_test, forecast_ARIMA))
    
    # RMSE in die Listen hinzufügen
    rmse_network_individual_per_technology.append(rmse_network_individual)
    rmse_network_panel_per_technology.append(rmse_network_panel)
    rmse_ARIMA_per_technology.append(rmse_ARIMA)
    
    if rmse_network_individual < rmse_ARIMA:
        better_model_per_technology_individual.append((tech, "Individual Network Model"))
    else:
        better_model_per_technology_individual.append((tech, "ARIMA Model"))


    if rmse_network_panel < rmse_ARIMA:
        better_model_per_technology_panel.append((tech, "Panel Network Model"))
    else:
        better_model_per_technology_panel.append((tech, "ARIMA Model"))

    cutoff_training=original_cutoff_training

average_rmse_network_individual = np.mean(rmse_network_individual_per_technology)
average_rmse_network_panel= np.mean(rmse_network_panel_per_technology)
average_rmse_ARIMA = np.mean(rmse_ARIMA_per_technology)
    
print(f"\nAverage RMSE for Network Model across all technologies: {average_rmse_network_individual:.4f}")
print(f"\nAverage RMSE for Panel Network Model across all technologies: {average_rmse_network_panel:.4f}")
print(f"Average RMSE for ARIMA Model across all technologies: {average_rmse_ARIMA:.4f}")
    
print("\nComparison of Best Model per Technology (individually estimated):")
for tech, model in better_model_per_technology_individual:
    print(f"{tech}: {model}")

print("\nComparison of Best Model per Technology (panel estimated):")
for tech, model in better_model_per_technology_panel:
    print(f"{tech}: {model}")


print("\nSum of unconditional_network_forecast_2030 for each technology (individually estimated):")
for tech, forecast_sum in unconditional_network_forecast_2030_sums:
    print(f"{tech}: {forecast_sum:.4f}")


print("\nSum of unconditional_network_forecast_2030 for each technology (panel estimated):")
for tech, forecast_sum in unconditional_panel_network_forecast_2030_sums:
    print(f"{tech}: {forecast_sum:.4f}")
