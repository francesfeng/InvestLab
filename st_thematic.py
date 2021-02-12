import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import SessionState
from streamlit.server.server import Server
import streamlit.report_thread as ReportThread

from streamlit_vega_lite import vega_lite_component, altair_component
pd.options.display.float_format = '{:,.0f}'.format

st.set_page_config(
     page_title="Thematic Playbook",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="auto",
 )



#---------------------------------------------------------------------------------------------------Import data
#@st.cache
def upload():
    price = pd.read_csv('to_upload/etf_wk.csv', index_col=0)
    etfs = pd.read_csv('to_upload/etf_ticker.csv')
    holdings = pd.read_csv('to_upload/holdings.csv', index_col=0)
    underlyings = pd.read_csv('to_upload/underlying_latest.csv', index_col=0)
    ratios = pd.read_csv('to_upload/ratios_ts_q.csv', index_col=0)
    type = pd.read_csv('to_upload/ratio_type.csv', index_col=0)
    stock_price = pd.read_csv('to_upload/price_wk.csv', index_col=0)


    price['Date'] = pd.to_datetime(price['Date'])
    ratios['Date'] = pd.to_datetime(ratios['Date'])
    underlyings['Date'] = pd.to_datetime(underlyings['Date'])
    stock_price['Date'] = pd.to_datetime(stock_price['Date'])

    listmap= {}
    listmap['Country'] = underlyings['Country'].unique()
    listmap['Industry'] = underlyings['Industry'].unique()
    listmap['Sector'] = underlyings['Sector'].unique()
    listmap['Size'] = underlyings['Size'].unique()

    return_des = underlyings[['Market Cap','1M(%)', '3M(%)', '6M(%)', 'YTD(%)']]
    return_min = return_des.min(axis=0)
    return_max = return_des.max(axis=0)
    return_bin = round((return_max - return_min)/100,0)
    return_pd = pd.concat([return_min, return_max, return_bin], axis=1)

    ratio_des = underlyings.iloc[:, 14:]
    ratio_min = ratio_des.min(axis=0)
    ratio_max = ratio_des.max(axis=0)
    ratio_bin = round((ratio_max - ratio_min)/100,2)
    ratio_pd = pd.concat([ratio_min, ratio_max, ratio_bin], axis=1)

    des = pd.concat([return_pd, ratio_pd], axis=0)
    des.columns = ['Min', 'Max', 'Bin']
    listmap['Ratio'] = des


    return (price, etfs, holdings, underlyings, ratios, type, listmap, stock_price)

#---------------------------------------------------------------------------------------------------Initiate data
etf_wk, etf_ticker, holdings, underlyings, ratios, type, listmap, stock_price = upload()
D0 = etf_wk['Date'].iloc[-1]
periods = pd.DataFrame(
         [D0 - np.timedelta64(3, "M"),
         D0 - np.timedelta64(6, "M"),
         D0 - np.timedelta64(1, "Y"),
         D0 - np.timedelta64(3, "Y"),
         D0 - np.timedelta64(5, "Y"),
         D0 - np.timedelta64(10, "Y"),
         etf_wk['Date'].iloc[0]
         ],
    index = ['3M', '6M', '1Y', '3Y', '5Y', '10Y', 'max']
)

etf_list = etf_ticker['Field'].tolist()

type_list = type['Type'].unique()
color_list = ['Sector', 'Industry', 'Country', 'Size']
ratio_list = np.array(type['Short Name'].unique())
size_list = np.append(['Market Cap', 'Size'], ratio_list)
filter_list = np.append(['Country', 'Sector', 'Industry', 'Size', 'Market Cap'], ratio_list)
ratio_slider = listmap['Ratio']


#---------------------------------------------------------------------------------------------------Select ETF

st.sidebar.title('Stock Universe')
etf_select = st.sidebar.multiselect(
     'Choose from Thematic ETFs',
     etf_list,(etf_list[1], etf_list[2])
)

etf_select_ticker = etf_ticker.loc[etf_ticker['Field'].isin(etf_select), 'Ticker']
etf_select_price = etf_wk[etf_wk['Ticker'].isin(etf_select_ticker)]

holding_select = holdings[holdings['ETF Ticker'].isin(etf_select_ticker)]

select_ticker = holding_select['Ticker'].unique()
underlyings_select = underlyings[underlyings['Ticker'].isin(select_ticker)]

port = SessionState.get(port_ticker=[])


#-------------------------------------------------------------------------------------------------Sidebar Filter

st.sidebar.title('Filters')
filter_num = st.sidebar.number_input(
    'Number of fiilters',
    0, 10, 3, 1
)


filters = {}
filter_val = {}
idx = {}
for i in range(0, filter_num):
    placeholder = st.sidebar.empty()
    filter_i = st.sidebar.selectbox(
        '',
        filter_list,
        i, key=i
    )

    placeholder.write("-----------------------")
    if filter_i in ['Sector', 'Industry', 'Country', 'Size']:
        filter_val_i = st.sidebar.multiselect(
            '',
            listmap[filter_i],
            key = i
        )

        if filter_val_i:
            idx[i] = underlyings_select[filter_i].isin(filter_val_i)


    else:
        local_list = ratio_slider.loc[filter_i].tolist()
        filter_val_i = st.sidebar.slider(
            '',
            local_list[0], local_list[1], (local_list[0], local_list[1]), local_list[2],
            key = i
        )
        if filter_val_i:
            idx[i] = (underlyings_select[filter_i] >= filter_val_i[0]) & (underlyings_select[filter_i] <= filter_val_i[1])




idx_full = np.array(np.ones(underlyings_select.shape[0]), dtype=bool)
for i in idx.keys():
    idx_full = idx_full & idx[i]


underlyings_filter = underlyings_select.loc[idx_full]


# --------------------------------------------------------------------------------------------------------Sidebar: Display

st.sidebar.title('Filter list')
if st.sidebar.button('Show results ( ' + str(len(underlyings_filter)) +' )'):
    st.sidebar.write(underlyings_filter[['Ticker', 'Company']].set_index('Ticker'))

filter_result_list = pd.concat([underlyings_filter['Ticker'],underlyings_filter['Ticker'] + ': ' + underlyings_filter['Company']], axis=1 )
filter_result_list.columns = ['Ticker', 'Company Display']

st.sidebar.title('Portfolio list: ' + str(len(port.port_ticker)))

port_manual_add = st.sidebar.multiselect(
    'Add stocks from filter list',
    filter_result_list.loc[~filter_result_list['Ticker'].isin(port.port_ticker),'Company Display'].tolist()
)

if st.sidebar.button('Add to portfolio'):
    port_add = filter_result_list.loc[filter_result_list['Company Display'].isin(port_manual_add), 'Ticker']
    idx = port_add.isin(port.port_ticker)
    list_add = port_add[~idx]
    port.port_ticker = np.append(port.port_ticker,list_add)


if len(port.port_ticker)>0:
    idx = underlyings_filter['Ticker'].isin(port.port_ticker)
    st.sidebar.table(underlyings_filter.loc[idx[:,np.newaxis], ['Ticker', 'Company']].set_index('Ticker'))
    if st.sidebar.button('Empty portfolio'):
        port.port_ticker = []

#add_criteria(s_add)



# -------------------------------------------------------------------------------------- ETF Performance

st.title('1. ETFs and underlying holdings analysis')
st.header('ETF Performance')

col_p1, col_p2, col_p3 = st.beta_columns([2,1,1])

val_p1 = col_p1.select_slider(
     'Period',
     options=['3M', '6M','1Y', '3Y', '5Y', '10Y', 'max']
)

val_p2 = col_p2.radio(
    'Type',
    ['Price', 'Total Return']
)

val_p3 = col_p3.radio(
    'Display',
    ['Absolute', 'Normalised']
)

def price_normalise(data, date_select = periods.loc['max',0], type = 2):
    date_min = data.groupby('Ticker')['Date'].min().tolist()
    date_min.append(date_select)
    max_date = max(date_min)
    data_sub = data[data['Date']>= max_date]
    fac = data_sub[data_sub.groupby('Ticker')['Date'].transform(min) == data_sub['Date']]
    if type == 1:

        #data.loc[data['Date'] == max_date1,['Ticker', 'Price']]
        fac['Factor1'] = 1/fac['Price']
        

        norm_fac = pd.merge(data_sub, fac[['Ticker', 'Factor1']], how='left', on='Ticker')
        norm_fac['Price_norm'] = norm_fac['Price'] * norm_fac['Factor1']

        norm_display = norm_fac[['Date', 'Ticker', 'Price_norm']]
        norm_display.columns = ['Date', 'Ticker', 'Price']
    else:
        #fac = data.loc[data['Date'] == max_date1,['Ticker', 'Price', 'Total Return']]
        fac['Factor1'] = 1/fac['Price']
        fac['Factor2'] = 1/fac['Total Return']

        norm_fac = pd.merge(data_sub, fac[['Ticker', 'Factor1', 'Factor2']], how='left', on='Ticker')
        norm_fac['Price_norm'] = norm_fac['Price'] * norm_fac['Factor1']
        norm_fac['TR_norm'] = norm_fac['Total Return'] * norm_fac['Factor2']

        norm_display = norm_fac[['Date', 'Ticker', 'Price_norm', 'TR_norm']]
        norm_display.columns = ['Date', 'Ticker', 'Price', 'Total Return']
    return(norm_display)

prd = periods.loc[val_p1,0]
etf_display = etf_select_price[etf_select_price['Date']> prd]

if val_p3 == 'Absolute':
    etf_display = etf_select_price[etf_select_price['Date']> prd]
else:
    etf_display = price_normalise(etf_select_price[etf_select_price['Date']> prd], prd)


# 3. Display Performance graph ---------------------------------

nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Date'], empty='none')

base = alt.Chart(etf_display).mark_line().encode(
    x='Date',
    color='Ticker:N',
    )
if val_p3 == 'Absolute':
    line = base.encode(y=val_p2 + ':Q')
else:
    line = base.encode(y=alt.Y(val_p2 + ':Q', axis=alt.Axis(format='%')))

selectors = alt.Chart(etf_display).mark_point().encode(
    x='Date',
    opacity=alt.value(0),
).add_selection(
    nearest
)

points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, val_p2+':Q', alt.value(' '))
    )

perform_graph = alt.layer(
    line, selectors, points, text
).properties(
    width=750,
    height = 500
    )
perform_graph

# 4. Holding data prep ---------------------------------
st.header('Holding analysis')

#industry_type = holding_select['Industry'].unique()
holding_filter = holding_select[holding_select['Ticker'].isin(underlyings_filter['Ticker'])]

col_h1, col_h2 = st.beta_columns(2)
holding_type = col_h1.selectbox(
    'View holdings by',
    ['Sector', 'Industry', 'Business Activity', 'Country', 'Market Cap']
)
#col_h2.empty()

def holding_data_display(industry, data):
    return data[data['Industry'] ==industry]

if holding_type == 'Market Cap':
    holding_type = 'Size'

if holding_type == 'Business Activity':
    holding_subtype = col_h2.selectbox(
        'Select an industry category:',
        listmap['Industry']
    )


holding_display_switch = st.radio("Dispaly", ('Stocks in filter results', 'Full ETF Holdings'), key = 1)

if holding_display_switch == 'Stocks in filter results':
    if holding_type == 'Business Activity':
        holding_display = holding_data_display(holding_subtype, holding_filter)
    else:
        holding_display =  holding_filter
else:
    if holding_type == 'Business Activity':
        holding_display = holding_data_display(holding_subtype, holding_select)
    else:
        holding_display = holding_select


# 4. Holding graph  ---------------------------------
@st.cache(suppress_st_warning=True)
def holding_graph():
    selector = alt.selection_single(encodings=['x','color'])

    holding_bar = alt.Chart(holding_display).mark_bar(
        align='left'
        #cornerRadiusTopLeft=3,
        #cornerRadiusTopRight=3
    ).encode(
        x='ETF Ticker:O',
        y='sum(Weight):Q',
        color=alt.condition(selector, holding_type+':N', alt.value('lightgray')),
        tooltip = ['ETF Ticker',holding_type, 'sum(Weight)'],
    ).add_selection(
        selector
    ).properties(
        width = 600
    )

    ticker_bar = alt.Chart(holding_display).mark_bar(align='left').encode(
        y = alt.Y('Company:O',sort='-x'),
        color = holding_type+':O',
        x = 'Weight',
        tooltip = ['ETF Ticker', 'Weight', 'Ticker', 'Company'],
    ).transform_filter(
        selector
    ).properties(
        #width='container'
    )
    return(holding_bar, ticker_bar)
graph1, graph2 = holding_graph()
graph1 & graph2


# ---------------------------------------------------------------------------------------------------Maxtrix Plot

st.header('Valuation Matrix')

underlying_display = underlyings_filter

col_u1, col_u2, col_u3, col_u4 = st.beta_columns(4)
val_u1 = col_u1.selectbox(
        'Y axis',
        type_list, index=0
)

val_u2 = col_u2.selectbox(
        'X axis',
        type_list,index=1
)

val_u3 = col_u3.selectbox(
        'Color by',
        color_list, index=2
)

val_u4 = col_u4.selectbox(
        'Size by',
        size_list, index=3
)

holding_display_switch2 = st.radio("Dispaly", ('Stocks in filter results', 'Full ETF Holdings'), key = 2)

if holding_display_switch2 == 'Stocks in filter results':
    underlying_display = underlyings_filter
else:
    underlying_display = underlyings_select

y_rows = type.loc[type['Type'] == val_u1,'Short Name']
x_rows = type.loc[type['Type'] == val_u2,'Short Name']

ratio_matrix = alt.Chart(underlying_display).mark_circle().encode(
                    alt.X(alt.repeat("column"), type='quantitative'),
                    alt.Y(alt.repeat("row"), type='quantitative'),
                    color=val_u3+':N',
                    size = val_u4+':Q',
                    tooltip=[ 'Ticker', 'Company']
                ).properties(
                    width=150,
                    height=150
                ).repeat(
                    row=y_rows.tolist(),
                    column=x_rows.tolist()
                ).interactive()

ratio_matrix

# --------------------------------------------------------------------------------------Time series plot
st.header('Historical Fundamentals')
col_t1, col_t2, col_t3 = st.beta_columns(3)


val_t1 = col_t1.selectbox(
        'Choose X',
        ratio_list
)
val_t1_transform = type.loc[type['Short Name'] == val_t1,'Alternative'].values[0]

val_t2 = col_t2.selectbox(
        'Choose Y',
        ratio_list[ratio_list!=val_t1]
)
val_t2_transform = type.loc[type['Short Name'] == val_t2,'Alternative'].values[0]


val_t3 = col_t3.radio('Display', (val_t1_transform, val_t2_transform))

holding_display_switch3 = st.radio("Dispaly", ('Stocks in filter results', 'Full ETF Holdings'), key = 3)

if holding_display_switch3 == 'Stocks in filter results':
    underlying_display = underlyings_filter
else:
    underlying_display = underlyings_select

# Display timeseries graph -------------------------------------------------------

selector = alt.selection_single(name="pivot_select", empty='all', fields=['Ticker'])
base = alt.Chart(underlying_display).properties(
        #width='container',
        #height=250
    ).add_selection(selector)

points = base.mark_circle(filled=True, size=200).encode(
        x=val_t2+':Q',
        y=val_t1+':Q',
        color=alt.condition(selector, 'Ticker:O', alt.value('lightgray'), legend=None),
        tooltip = ['Ticker', 'Company']
    ).properties(
        width = 'container'
    )

ratios_display = ratios[['Ticker', 'Date', val_t1_transform, val_t2_transform]]
timeseries = alt.Chart(ratios_display).mark_bar().encode(
        x='Date:T',
        y=val_t3+':Q',
        tooltip = ['Ticker:O', 'Company:O', 'Date:T',val_t3]
    ).add_selection(
        selector
    ).transform_filter(
        'datum.Ticker == pivot_select.Ticker'
    ).properties(
        width='container',
        #height=500
    )
points | timeseries

# -------------------------------------------------------------------------------------------- Peer Comparison Graph

st.header('Pair relationships')

col_c0, col_c4, col_c5 = st.beta_columns(3)
val_c0 = col_c0.selectbox('Y axis',ratio_list)
val_c4 = col_c4.selectbox('Colour by', color_list)
val_c5 = col_c5.selectbox('Size by', size_list, key=2)

col_c1, col_c2, col_c3= st.beta_columns(3)
val_c1 = col_c1.selectbox('x1 ratio', ratio_list[ratio_list!=val_c0])
val_c2 = col_c2.selectbox('x2 ratio', ratio_list[ (ratio_list!=val_c0) & (ratio_list!=val_c1)])
val_c3 = col_c3.selectbox('x2 ratio', ratio_list[ (ratio_list!=val_c0) & (ratio_list!=val_c1)& (ratio_list!=val_c2)])

holding_display_switch4 = st.radio("Dispaly", ('Stocks in filter results', 'Full ETF Holdings'), key = 4)

if holding_display_switch4 == 'Stocks in filter results':
    underlying_display = underlyings_filter
else:
    underlying_display = underlyings_select


@st.cache(suppress_st_warning=True)
def pair_scatter(legend_flag):
    brush = alt.selection(type='interval', resolve='global')

    base =  alt.Chart(underlying_display).mark_point().encode(
            y=val_c0+':Q',
            color=alt.condition(brush, 'Sector:N', alt.ColorValue('gray'), legend=legend_flag),
            size = 'Size:N',
            tooltip=[ 'Ticker', 'Company:O', 'Sector:N']
        ).add_selection(
            brush
        ).properties(width='container')

    base1 = base.encode(x=val_c1+':Q')
    base2 = base.encode(x=val_c2+':Q')
    base3 = base.encode(x=val_c3+':Q')

    return (base1 , base2 , base3)

base1, base2, base3 = pair_scatter(None)
base1 | base2 | base3
# Try -------------------------------------------------
#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
#def pair_try():
#    brush = alt.selection(type='interval')

# Define the base chart, with the common parts of the
# background and highlights
#    base = alt.Chart().mark_point().encode(
#        x=alt.X(alt.repeat('column'), type='quantitative'),
#        y=val_c0+':Q',
#        ).properties(
#            width=160,
#            height=130
#        )

#    background = base.encode(
#    color=alt.value('#ddd')
#    ).add_selection(brush)

#    highlight = base.transform_filter(brush)

#    return alt.layer(
#    background,
#    highlight,
#    data=underlying_display
#    ).repeat(column=[val_c1, val_c2, val_c3])

#a = pair_try()
#b = altair_component(pair_try())
#st.write(b)

# --------------------------------------------------------------------------------------- Portfolio built
st.title('2. Portfolio Contruction')
st.subheader('Drag across the graph to select stocks to be added to the portfolio')
st.write('--------------------------------')

col_b1,col_b2, col_b3, col_b4 = st.beta_columns([1,1,1,1])

val_b1 = col_b1.selectbox(
        'Y axis:',
        size_list,2
        )

val_b2 = col_b2.selectbox(
        'X axis:',
        size_list,5
    )

val_b3 = col_b3.selectbox(
    'Rank fundamentals',
    size_list,6
)

val_b4 = col_b4.select_slider(
    'Filter by',
    ['Bottem percentile', 'Middle quantile', 'Top quantile']

)

@st.cache(suppress_st_warning=True)
def single_scatter(y, x):
    brush = alt.selection(type='interval', resolve='global')

    return alt.Chart(underlying_display).mark_point().encode(
                y= y + ':Q',
                x = x + ':Q',
                color=alt.condition(brush, 'Sector:N', alt.ColorValue('gray'), legend=None),
                tooltip=[ 'Ticker', 'Company:O', 'Sector:N']
            ).add_selection(
                brush
            ).properties(
                width=500
            )
@st.cache
def rank_bar(y):
    temp = underlying_display[['Ticker', 'Company',y]]
    brush3 = alt.selection(type='interval', resolve='global',encodings=['x'])
    bar = alt.Chart(temp).mark_bar().encode(
        x=alt.X('Ticker:O',sort='-y'),
        y=y+':Q',
        color = alt.condition(brush3, 'Ticker:N', alt.ColorValue('grey'), legend=None)
        )
    rule = alt.Chart(temp).mark_rule(color='yellow').encode(
        y='mean('+y+'):Q'
        )
    return (bar.add_selection(
                brush3
                ).properties(
                    width=500
                ))
col_b4, col_b5 = st.beta_columns(2)
with col_b4:
    s_select = altair_component(single_scatter(val_b1, val_b2))

with col_b5:
    t_select = altair_component(rank_bar(val_b3))


#-----------------------------------------------------------------------------------------------Port select display
def add_to_session_port(data, add_to):
    idx = data.isin(add_to)
    list_add = data[~idx]
    add_to = np.append(add_to,list_add)
    return(add_to, list_add)

if s_select:
    lower1 = s_select[val_b1][0]
    upper1 = s_select[val_b1][1]
    lower2 = s_select[val_b2][0]
    upper2 = s_select[val_b2][1]

    port_select1 = underlying_display[(underlying_display[val_b1] >= lower1) & (underlying_display[val_b1] <= upper1) &
                                    (underlying_display[val_b2] >= lower2) & (underlying_display[val_b2] <= upper2)
                                    ]
else:
    port_select1 = underlying_display

if t_select:
    #st.write(t_select['Ticker'][:]
    t_list = t_select['Ticker']
    port_select2 = underlying_display[underlying_display['Ticker'].isin(t_list)]
else:
    port_select2 = underlying_display
    #st.write(port_select2)

col_f1, col_f2, col_f3, col_f4 = st.beta_columns(4)

if s_select:
    col_f1.write('Selected stocks: ' + str(len(port_select1)))
else:
    col_f1.write('All filtered stocks: ' + str(len(underlying_display)))


if col_f2.button('Add all to portfolio', key=1):
    port.port_ticker, list_add1 = add_to_session_port(port_select1['Ticker'], port.port_ticker)
    st.info(str(len(list_add1)) + ' stocks are added, ' + str(len(port_select1) - len(list_add1)) + ' already exist in the portfolio' )

if t_select:
    col_f3.write('Selected stocks: ' + str(len(port_select2)))
else:
    col_f3.write('All filtered stocks: ' + str(len(port_select2)))

if col_f4.button('Add all to portfolio', key=2):
    port.port_ticker, list_add2 = add_to_session_port(port_select2['Ticker'], port.port_ticker)
    st.info(str(len(list_add2)) + ' stocks are added, ' + str(len(port_select2) - len(list_add2)) + ' already exist in the portfolio' )


col_d1, col_d2 = st.beta_columns(2)

col_d1.write(port_select1[['Ticker', 'Company', val_b1, val_b2]])
col_d2.write(port_select2[['Ticker', 'Company', val_b3]])

#----------------------------------------------------------------------------------------------------------Weight
port_struc = underlyings[underlyings['Ticker'].isin(port.port_ticker)]
port_struc.index = [x for x in range(1, len(port_struc)+1)]
col_list = ['Ticker', 'Company', 'Industry','Weight']

col_a1, col_a2 = st.beta_columns(2)
col_a1.header('Portfolio allocation')
col_a2.subheader('')

if col_a2.checkbox('Show my portfolio in full view'):
    col_w1, col_w2, col_w3, col_w4 = st.beta_columns(4)
    val_w1 = col_w1.selectbox(
            'Weight by',
            ['Market Cap', 'Equal Weight', 'Fundamental Weight']
             )
    val_w2 = col_w2.selectbox(
            'Dividend reinvest',
            ['Yes', 'No']
    )

    val_v3 = col_w3.selectbox(
            'Rebalance',
            ['Quarterly', 'Monthly', 'Annual']
            )

    val_v4 = col_w4.selectbox(
            'Add columns',
            size_list
    )
    #st.sidebar.table(underlyings_filter.loc[idx[:,np.newaxis], ['Ticker', 'Company']])

    if val_w1 == 'Market Cap':
        weightsum = port_struc['Market Cap'].sum()
        weight = port_struc['Market Cap'] / weightsum
    elif val_w1 == 'Equal Weight':
        weight = 1/len(port_struc)
    else:
        weight = 1

    port_struc['Weight'] = weight

    if ~(val_v4 in col_list):
        col_list = np.append(col_list, val_v4)

    st.table(port_struc[col_list])

    port_stock_price = stock_price[ stock_price['Ticker'].isin(port.port_ticker)]
    port_stock_pivot = port_stock_price.pivot_table(index='Date', columns = 'Ticker', values='CLOSE')

    port_stock_pivot.fillna(method='ffill', inplace=True)

    port_stock_p = port_stock_pivot[port_struc['Ticker']]
    port_stock_valid = port_stock_p[~port_stock_p.isna().any(axis=1)]

    index_cal = port_stock_valid.values * weight.values
    index_pd = pd.DataFrame(np.sum(index_cal, axis=1), index = port_stock_valid.index)
    index_pd.reset_index(inplace=True)
    index_pd['Ticker'] = 'Portfolio'
    index_pd.columns = ['Date', 'Price', 'Ticker']

    port_all = pd.concat([index_pd, etf_select_price.iloc[:,:3]], axis=0, ignore_index=True)

    port_norm = price_normalise(port_all, periods.loc['max',0], type=1)

    #val_p1 = col_p1.select_slider(
    #     'Period',
    #     options=['3M', '6M','1Y', '3Y', '5Y', '10Y', 'max']
    #)
    #st.write(port_stock_valid)
    base1 = alt.Chart(index_pd).mark_line().encode(
        x='Date',
        y = 'Price:Q',
        color='Ticker:N',
        )
    base1

    base = alt.Chart(port_norm).mark_line().encode(
        x='Date',
        y = 'Price:Q',
        color='Ticker:N',
        )
    base

    st.write(port_norm)
    st.write(port_all)
