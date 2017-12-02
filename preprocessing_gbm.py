import numpy as np
import pandas as pd



IDIR = '../data/'


print('loading prior')
df_priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
df_train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
df_orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'object',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
df_products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

print('priors {}: {}'.format(df_priors.shape, ', '.join(df_priors.columns)))
print('orders {}: {}'.format(df_orders.shape, ', '.join(df_orders.columns)))
print('train {}: {}'.format(df_train.shape, ', '.join(df_train.columns)))


###########################
# COMPUTE PRODUCT FEATURES
###########################
print('computing product features (num orders, num reorders, reorder rate)')
num_prior_orders = len(df_priors.order_id.unique())
df_prods_temp = pd.DataFrame()
df_prods_temp['orders'] = df_priors.groupby(df_priors.product_id).size().astype(np.int32)
df_prods_temp['reorders'] = df_priors['reordered'].groupby(df_priors.product_id).sum().astype(np.float32)
df_prods_temp['reorder_rate'] = (df_prods_temp.reorders / df_prods_temp.orders).astype(np.float32)
df_prods_temp['prob_purchase'] = (df_prods_temp['orders'] / num_prior_orders).astype(np.float32)
df_products = df_products.join(df_prods_temp, on='product_id')
df_products.set_index('product_id', drop=False, inplace=True)
del df_prods_temp

df_products.to_csv('df_products.csv')


################################
# AISLE AND DEPARTMENT FEATURES
################################
# aisle 
df_aisles = pd.DataFrame()
df_aisles['aisle_id'] = df_products.aisle_id.unique()
df_aisles.set_index('aisle_id', drop=False, inplace=True)
df_aisle_temp = pd.DataFrame()
df_prods_priors = df_priors.join(df_products, on='product_id', rsuffix='_')
df_aisle_temp['aisle_orders'] = df_prods_priors.groupby(df_prods_priors.aisle_id).size().astype(np.int32)
df_aisle_temp['aisle_reorders'] = df_prods_priors['reordered'].groupby(df_prods_priors.aisle_id).sum().astype(np.float32)
df_aisle_temp['aisle_reorder_rate'] = (df_aisle_temp.aisle_reorders / df_aisle_temp.aisle_orders).astype(np.float32)
df_aisle_temp['aisle_prob_purchase'] = (df_aisle_temp['aisle_orders'] / num_prior_orders).astype(np.float32)
# department
df_department = pd.DataFrame()
df_department['department_id'] = df_products.department_id.unique()
df_department.set_index('department_id', drop=False, inplace=True)
df_deparment_temp = pd.DataFrame()
df_deparment_temp['department_orders'] = df_prods_priors.groupby(df_prods_priors.department_id).size().astype(np.int32)
df_deparment_temp['department_reorders'] = df_prods_priors['reordered'].groupby(df_prods_priors.department_id).sum().astype(np.float32)
df_deparment_temp['department_reorder_rate'] = (df_deparment_temp.department_reorders / df_deparment_temp.department_orders).astype(np.float32)
df_deparment_temp['department_prob_purchase'] = (df_deparment_temp['department_orders'] / num_prior_orders).astype(np.float32)
# join
df_aisles = df_aisles.join(df_aisle_temp, on='aisle_id', rsuffix='_aisle_')
df_department = df_department.join(df_deparment_temp, on='department_id', rsuffix='_department_')
del df_aisle_temp
del df_deparment_temp
del df_prods_priors
df_aisles.to_csv('df_aisles.csv')
df_department.to_csv('df_department.csv')



####################################
# ADD ORDER INFO TO PRIOR DATA FRAME
####################################
print('add order info to priors')
df_orders.set_index('order_id', inplace=True, drop=False)
df_priors = df_priors.join(df_orders, on='order_id', rsuffix='_')
df_priors.drop('order_id_', inplace=True, axis=1)


####################################
# COMPUTE USER FEATURES
####################################
print('computing user features (average days between orders, num orders, total purchased items, all different products ids bought, num diff items, average basket)')
df_usr_temp = pd.DataFrame()
df_usr_temp['average_days_between_orders'] = df_orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
df_usr_temp['mean_hour_purchase'] = df_orders.groupby('user_id')['order_hour_of_day'].mean().astype(np.float32)
df_usr_temp['median_hour_purchase'] = df_orders.groupby('user_id')['order_hour_of_day'].median().astype(np.float32)
df_usr_temp['nb_orders'] = df_orders.groupby('user_id').size().astype(np.int16)
# About Day of the week
most_frequent_day = []
n_orders_most_frequent_day = []
for user_dow in df_orders.groupby('user_id')['order_dow']:
    most_frequent_day.append(user_dow[1].value_counts().idxmax())
    n_orders_most_frequent_day.append(user_dow[1].value_counts().max())
df_usr_temp['most_frequent_day'] = np.array(most_frequent_day).astype(np.int8)
df_usr_temp['n_orders_most_frequent_day'] = np.array(n_orders_most_frequent_day).astype(np.int16)
df_usr_temp['prop_orders_most_frequent_day'] = (df_usr_temp['n_orders_most_frequent_day'] / df_usr_temp['nb_orders']).astype(np.float32)
dow_last_prior_purchase = []
for user in df_orders[df_orders.eval_set == 'prior'].groupby('user_id'):
    dow_last_prior_purchase.append(user[1][user[1].order_number == user[1].order_number.max()].order_dow.values[0])
df_usr_temp['dow_last_prior_purchase'] = np.array(dow_last_prior_purchase).astype(np.int8)

df_users = pd.DataFrame()
df_users['total_items'] = df_priors.groupby('user_id').size().astype(np.int16)
df_users['all_products'] = df_priors.groupby('user_id')['product_id'].apply(set)
df_users['total_distinct_items'] = (df_users.all_products.map(len)).astype(np.int16)

df_users = df_users.join(df_usr_temp)
del df_usr_temp
df_users['average_basket'] = (df_users.total_items / df_users.nb_orders).astype(np.float32)
print('user f', df_users.shape)
df_users.to_csv('df_users.csv')


########################################
# COMPUTE USER-PRODUCT AND PRODUCT CANDIDATE FEATURES
########################################

print('compute candidate features...')
df_priors['user_product'] = df_priors.product_id.astype(np.int64) + df_priors.user_id.astype(np.int64) * 100000 

d_user_product = dict() #tuples user-product(times_candidate, times_reordered, times_candidate_after_purchase_last_order, times_reordered_next_order)
n_users = len(df_priors.user_id.unique())
i = 0
for user,group in df_priors.groupby('user_id'):
    total_orders = group.order_number.max()
    for product_id in group.product_id.unique():
        product_orders = group[group.product_id == product_id]
        first_order = product_orders.order_number.min()
        times_candidate = total_orders - first_order
        times_reorder = len(product_orders) - 1
        next_order_candidate = len(group[(group.product_id == product_id) & (group.order_number < total_orders)]) 
        times_reorder_next_order = np.sum(np.ediff1d(product_orders.sort_values('order_number').order_number.values) == 1)
        user_product = product_id.astype(np.int64) + user.astype(np.int64) * 100000
        d_user_product[user_product] = (times_candidate, times_reorder, next_order_candidate, times_reorder_next_order)
    i += 1
    if i % 10000 == 0:
        print(str(i) + "-" + str(n_users))
      
      
d_products = {} #tuples product (times_candidate, times_reordered, times_candidate_after_purchase_last_order, times_reordered_next_order, diff_users)
for user_product in d_user_product:
    product_id = user_product % 100000
    times_candidate = d_user_product[user_product][0]
    times_reordered = d_user_product[user_product][1]
    times_candidate_next_order = d_user_product[user_product][2]
    times_reordered_next_order = d_user_product[user_product][3]
    if product_id in d_products:
        d_products[product_id] = (d_products[product_id][0] + times_candidate,
                       d_products[product_id][1] + times_reordered,
                        d_products[product_id][2] + times_candidate_next_order,
                        d_products[product_id][3] + times_reordered_next_order,
                        d_products[product_id][4] + 1)
    else:
        d_products[product_id] = (times_candidate, times_reordered, times_candidate_next_order, times_reordered_next_order, 1)
     
     
df_userXproduct_candidate = pd.DataFrame.from_dict(d_user_product, orient='index')
df_userXproduct_candidate.columns = ['UP_times_candidate', 
                                     'UP_times_reordered', 
                                     'UP_times_candidate_next_order', 
                                     'UP_times_reordered_next_order']
del d_user_product
df_userXproduct_candidate.UP_times_candidate = df_userXproduct_candidate.UP_times_candidate.astype(np.int16)
df_userXproduct_candidate.UP_times_reordered = df_userXproduct_candidate.UP_times_reordered.astype(np.int16)
df_userXproduct_candidate.UP_times_candidate_next_order = df_userXproduct_candidate.UP_times_candidate_next_order.astype(np.int16)
df_userXproduct_candidate.UP_times_reordered_next_order = df_userXproduct_candidate.UP_times_reordered_next_order.astype(np.int16)
df_userXproduct_candidate.to_csv('df_userXproduct_candidate.csv')
    

df_product_candidate = pd.DataFrame.from_dict(d_products, orient='index')
df_product_candidate.columns = ['times_candidate', 
                                     'times_reordered', 
                                     'times_candidate_next_order', 
                                     'times_reordered_next_order',
                                     'diff_users']

del d_products
df_product_candidate.times_candidate = df_product_candidate.times_candidate.astype(np.int32)
df_product_candidate.times_reordered = df_product_candidate.times_reordered.astype(np.int32)
df_product_candidate.times_candidate_next_order = df_product_candidate.times_candidate_next_order.astype(np.int32)
df_product_candidate.times_reordered_next_order = df_product_candidate.times_reordered_next_order.astype(np.int32)
df_product_candidate.diff_users = df_product_candidate.diff_users.astype(np.int32)
df_product_candidate.to_csv('df_product_candidate.csv')



####################################
# COMPUTE USER-PRODUCT FEATURES
####################################
from collections import Counter

print('compute userXproduct features - this is long...')
#df_priors['user_product'] = df_priors.product_id.astype(np.int64) + df_priors.user_id.astype(np.int64) * 100000 # Check there is no collision

# dictionary of tuples user-product 
d = dict()  # (num orders, (last order number, last_order_id, sum_pos), sum_pos_car)
for row in df_priors.itertuples():
    user_product = row.user_product
    if user_product not in d:
        d[user_product] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[user_product] = (d[user_product][0] + 1,
                max(d[user_product][1], (row.order_number, row.order_id)),
                d[user_product][2] + row.add_to_cart_order)
                
# Second set of features
#mean_num_days_previous_orders, mean_hour, std_hour, most_common_day, ocurrences_most_common_day, proportion_most_common_day
i = 0
d_user_actual_day = {}
d2 = {}
for user,user_group in df_priors.groupby('user_id'):
    actual_day = 0
    actual_order = 1
    user_products = []
    for row in user_group.sort_values('order_number', ascending=True).itertuples():
        user_product = row.user_product
        user_products.append(user_product)
        if row.order_number != actual_order:
            actual_order += 1
            actual_day += row.days_since_prior_order
        if user_product not in d2:
            d2[user_product] = {}
            d2[user_product]['order_days'] = [actual_day]
            d2[user_product]['hours'] = [row.order_hour_of_day]
            d2[user_product]['days_of_week'] = [row.order_dow]
        else:
            d2[user_product]['order_days'].append(actual_day)
            d2[user_product]['hours'].append(row.order_hour_of_day)
            d2[user_product]['days_of_week'].append(row.order_dow)
    for user_product in list(set(user_products)):
        if len(d2[user_product]['order_days']) > 1:
            mean_freq_days_order = np.mean(np.ediff1d(d2[user_product]['order_days'])).astype(np.float32)
            median_freq_days_order = np.median(np.ediff1d(d2[user_product]['order_days'])).astype(np.float32)
            only_one_order = False
        else:
            mean_freq_days_order = np.float32(0.0)
            median_freq_days_order = np.float32(0.0)
            only_one_order = True
        day_last_order = np.int16(d2[user_product]['order_days'][-1])
        mean_hours = np.mean(d2[user_product]['hours']).astype(np.float32)
        std_hour = np.std(d2[user_product]['hours']).astype(np.float32)
        hour_last_order = np.int8(d2[user_product]['hours'][-1])
        counter=Counter(d2[user_product]['days_of_week'])
        most_common_day_of_week = np.int8(counter.most_common(1)[0][0])
        occurences_most_common_day_of_week = np.int16(counter.most_common(1)[0][1]) 
        day_of_week_last_order = np.int8(d2[user_product]['days_of_week'][-1])   
        d2[user_product] = (mean_freq_days_order, 
                           median_freq_days_order,
                           only_one_order,
                           day_last_order,
                           mean_hours,
                           std_hour,
                           hour_last_order,
                           most_common_day_of_week,
                           occurences_most_common_day_of_week,
                           day_of_week_last_order)
    d_user_actual_day[user] = actual_day
    if i % 10000 == 0:
        print(i)
    i += 1
    
    
# Join dictionaries 
for user_product in d:
    d[user_product] = (d[user_product][0], d[user_product][1], d[user_product][2], 
                       d2[user_product][0],
                       d2[user_product][1],
                       d2[user_product][2],
                       d2[user_product][3],
                       d2[user_product][4],
                       d2[user_product][5],
                       d2[user_product][6],
                       d2[user_product][7],
                       d2[user_product][8],
                       d2[user_product][9])
del d2

# Create dataframe 
print('to dataframe (less memory)')
df_userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d
df_userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart', 
                           'mean_freq_days_order', 'median_freq_days_order',
                           'only_one_order', 'day_last_order',
                           'mean_hours', 'std_hour',
                           'hour_last_order', 'most_common_day_of_week',
                           'occurences_most_common_day_of_week', 'day_of_week_last_order']
df_userXproduct.nb_orders = df_userXproduct.nb_orders.astype(np.int16)
df_userXproduct.last_order_id = df_userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
df_userXproduct.sum_pos_in_cart = df_userXproduct.sum_pos_in_cart.astype(np.int16)
df_userXproduct.mean_freq_days_order = df_userXproduct.mean_freq_days_order.astype(np.float32)
df_userXproduct.median_freq_days_order = df_userXproduct.median_freq_days_order.astype(np.float32)
df_userXproduct.only_one_order = df_userXproduct.only_one_order.astype(np.bool_)
df_userXproduct.day_last_order = df_userXproduct.day_last_order.astype(np.int16)
df_userXproduct.mean_hours = df_userXproduct.mean_hours.astype(np.float32)
df_userXproduct.std_hour = df_userXproduct.std_hour.astype(np.float32)
df_userXproduct.hour_last_order = df_userXproduct.hour_last_order.astype(np.int8)
df_userXproduct.most_common_day_of_week = df_userXproduct.most_common_day_of_week.astype(np.int8)
df_userXproduct.occurences_most_common_day_of_week = df_userXproduct.occurences_most_common_day_of_week.astype(np.int8)
df_userXproduct.day_of_week_last_order = df_userXproduct.day_of_week_last_order.astype(np.int8)
df_userXproduct['proportion_most_common_day'] = (df_userXproduct.occurences_most_common_day_of_week / df_userXproduct.nb_orders).astype(np.float32)
print('user X product f', len(df_userXproduct))

del df_priors
df_userXproduct.to_csv('df_userXproduct.csv')

####################################
# TRAIN/TEST ORDERS
####################################
### train / test orders ###
print('split orders : train, test')
df_test_orders = df_orders[df_orders.eval_set == 'test']
df_train_orders = df_orders[df_orders.eval_set == 'train']
df_train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

####################################
# BUILD LIST OF CANDIDATE PRODUCTS TO REORDER WITH FEATURES
####################################
### build list of candidate products to reorder, with features ###
def features(df_selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    for row in df_selected_orders.itertuples():
        i+=1
        if i%10000 == 0: print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = df_users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in df_train.index for product in user_products]
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    print('user related features')
    df['user_id'] = df.order_id.map(df_orders.user_id)
    df['user_total_orders'] = df.user_id.map(df_users.nb_orders)
    df['user_total_items'] = df.user_id.map(df_users.total_items)
    df['total_distinct_items'] = df.user_id.map(df_users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(df_users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(df_users.average_basket)
    df['mean_hour_purchase'] = df.user_id.map(df_users.mean_hour_purchase) # NEW
    df['median_hour_purchase'] = df.user_id.map(df_users.median_hour_purchase) # NEW
    df['most_frequent_day'] = df.user_id.map(df_users.most_frequent_day) # NEW
    df['n_orders_most_frequent_day'] = df.user_id.map(df_users.n_orders_most_frequent_day) # NEW
    df['prop_orders_most_frequent_day'] = df.user_id.map(df_users.prop_orders_most_frequent_day) # NEW
    df['dow_last_prior_purchase'] = df.user_id.map(df_users.dow_last_prior_purchase) # NEW
    df['day_last_order'] = df['user_id'].map(d_user_actual_day).astype(np.int16) # NEW (day count, not day of the week)
    
    print('order related features')
    df['order_dow'] = df.order_id.map(df_orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(df_orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(df_orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    df['delta_hour_vs_average'] = abs(df.order_hour_of_day - df.mean_hour_purchase).map(lambda x: min(x, 24-x)).astype(np.float32)
    df['same_day_most_common_day'] = (df['most_frequent_day'] == df.order_dow) # NEW
    df['same_day_last_order_day'] = (df['dow_last_prior_purchase'] == df.order_dow) # NEW
    df['current_day'] = (df.day_last_order + df.days_since_prior_order).astype(np.int16) # NEW (day count, not day of the week)
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(df_products.aisle_id)
    df['department_id'] = df.product_id.map(df_products.department_id)
    df['product_orders'] = df.product_id.map(df_products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(df_products.reorders)
    df['product_reorder_rate'] = df.product_id.map(df_products.reorder_rate)
    df['prob_purchase'] = df.product_id.map(df_products.prob_purchase) # NEW
    # Features about product is candidate
    df['times_candidate'] = df.product_id.map(df_product_candidate.times_candidate) # NEW
    df['times_reordered'] = df.product_id.map(df_product_candidate.times_reordered) # NEW
    df['times_candidate_next_order'] = df.product_id.map(df_product_candidate.times_candidate_next_order) # NEW
    df['times_reordered_next_order'] = df.product_id.map(df_product_candidate.times_reordered_next_order) # NEW
    
    print('aisle related features')
    df['aisle_orders'] = df.aisle_id.map(df_aisles.aisle_orders).astype(np.int32)
    df['aisle_reorders'] = df.aisle_id.map(df_aisles.aisle_reorders).astype(np.int32)
    df['aisle_reorder_rate'] = df.aisle_id.map(df_aisles.aisle_reorder_rate).astype(np.float32)
    df['aisle_prob_purchase'] = df.aisle_id.map(df_aisles.aisle_prob_purchase).astype(np.float32)
    
    print('department related features')
    df['department_orders'] = df.department_id.map(df_department.department_orders).astype(np.int32)
    df['department_reorders'] = df.department_id.map(df_department.department_reorders).astype(np.int32)
    df['department_reorder_rate'] = df.department_id.map(df_department.department_reorder_rate).astype(np.float32)
    df['department_prob_purchase'] = df.department_id.map(df_department.department_prob_purchase).astype(np.float32)
    

    print('user_X_product related features')
    df['user_product'] = df.user_id.astype(np.int64) * 100000 + df.product_id.astype(np.int64)
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.user_product.map(df_userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.user_product.map(df_userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.user_product.map(df_userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(df_orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(df_orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
    df['UP_mean_freq_days_order'] = df.user_product.map(df_userXproduct.mean_freq_days_order) # NEW
    df['UP_median_freq_days_order'] = df.user_product.map(df_userXproduct.median_freq_days_order) # NEW
    df['UP_only_one_order'] = df.user_product.map(df_userXproduct.only_one_order) # NEW
    df['UP_day_last_order'] = df.user_product.map(df_userXproduct.day_last_order) # NEW
    # Here add the difference with the current day (), and this with the mean (use dictionary)
    df['UP_days_from_last_purchase'] = (df.current_day - df.UP_day_last_order).astype(np.int16) # NEW
    df['UP_difference_last_purchase_and_mean'] = np.abs(df.UP_days_from_last_purchase - df.UP_median_freq_days_order).astype(np.float32) # NEW
    df['UP_days_from_last_purchase_ratio'] = df.current_day / df.UP_day_last_order # NEW
    df['UP_mean_hours'] = df.user_product.map(df_userXproduct.mean_hours) # NEW
    df['UP_delta_hour_vs_average_hour'] = abs(df.order_hour_of_day - df.UP_mean_hours).map(lambda x: min(x, 24-x)).astype(np.float32) # NEW
    df['UP_std_hour'] = df.user_product.map(df_userXproduct.std_hour) # NEW
    df['UP_most_common_day_of_week'] = df.user_product.map(df_userXproduct.most_common_day_of_week) # NEW
    df['UP_occurences_most_common_day_of_week'] = df.user_product.map(df_userXproduct.occurences_most_common_day_of_week) # NEW
    df['UP_proportion_occurences_most_common_day_of_week'] = df.user_product.map(df_userXproduct.proportion_most_common_day) # NEW
    #df['UP_day_of_week_last_order'] = df.user_product.map(df_userXproduct.day_of_week_last_order) # NEW
    df['UP_same_day_most_common_day'] = (df['UP_most_common_day_of_week'] == df.order_dow) # NEW
    df['UP_day_of_week_last_order'] = df.UP_last_order_id.map(df_orders.order_dow).astype(np.int8) # NEW
    df['UP_same_day_last_order'] = (df['UP_day_of_week_last_order'] == df.order_dow) # NEW     
    # Features about the number of times is candidate
    df['UP_times_candidate'] = df.user_product.map(df_userXproduct_candidate.UP_times_candidate) # NEW
    df['UP_times_reordered'] = df.user_product.map(df_userXproduct_candidate.UP_times_reordered) # NEW
    df['UP_times_candidate_next_order'] = df.user_product.map(df_userXproduct_candidate.UP_times_candidate_next_order) # NEW
    df['UP_times_reordered_next_order'] = df.user_product.map(df_userXproduct_candidate.UP_times_reordered_next_order) # NEW
 
    df.drop(['UP_last_order_id', 'user_product', 'day_last_order', 'UP_day_last_order'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)
    

df_train, labels = features(df_train_orders, labels_given=True)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last',
       'n_orders_most_frequent_day', 'prop_orders_most_frequent_day',  # NEW user features
       'same_day_most_common_day', 'same_day_last_order_day', # NEW order feautres (order-user)
       'prob_purchase']


# Save/Load training dataframe in csv
df_train.to_csv('df_train.csv')
# Save pickle of labels
import pickle
with open("pickles/labels.pickle", 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL) 
del df_train
    
    
### build candidates list for test ###
df_test, _ = features(df_test_orders)
df_test.to_csv('df_test.csv')

    
    
    








