import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:

    def __init__(self, *args):

        olist = Olist()
        if args:
            self.data = olist.get_data(args[0])
            self.matching_table = olist.get_matching_table(args[0])
            self.order = Order(args[0])
        else:
            self.data = olist.get_data()
            self.matching_table = olist.get_matching_table()
            self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
       'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers']
        return sellers[['seller_id', 'seller_city', 'seller_state']]

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """
        def delay(x):
            if x > 0:
                return abs(x)
            return 0

        orders = self.data['orders'].query("order_status=='delivered'").copy()
        order_items = self.data['order_items'][['order_id', 'shipping_limit_date']].copy()
        values = self.matching_table[['order_id','seller_id']].merge(orders, on='order_id', how='inner')
        values = values.merge(order_items, on='order_id', how='inner')
        colum = ['order_purchase_timestamp', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'shipping_limit_date']
        for i in colum:
            values[i] = pd.to_datetime(values[i], format='%Y-%m-%d %H:%M:%S')
        values['wait_time'] = (values['order_delivered_customer_date'] - values['order_purchase_timestamp']) / np.timedelta64(24, 'h')
        values['difff'] = (values['order_delivered_carrier_date'] - values['shipping_limit_date']) / np.timedelta64(24, 'h')
        values["delay_to_carrier"] = values["difff"].map(delay) # --> Series([0, 1, 1, 0, 0, 1 ...])
        values = values[['seller_id', 'wait_time', 'delay_to_carrier']].dropna()
        return values.groupby('seller_id', as_index=False).mean()

    def get_active_dates(self):
        """
        Returns a DataFrame with:
       'seller_id', 'date_first_sale', 'date_last_sale'
        """
        orders = self.data['orders'][['order_id','order_approved_at']].copy()
        values = self.matching_table[['order_id','seller_id']].merge(orders, on='order_id', how='right')
        colum = ['order_approved_at']
        for i in colum:
            values[i] = pd.to_datetime(values[i], format='%Y-%m-%d %H:%M:%S')
        # values['date_last_sale'] = values.groupby('seller_id').max().reset_index()['order_approved_at']
        values['date_first_sale'] = values['order_approved_at']
        values['date_last_sale'] = values['order_approved_at']
        values = values.groupby('seller_id').agg({'date_first_sale': 'min', 'date_last_sale':'max'}).reset_index()
        return values[['seller_id', 'date_first_sale', 'date_last_sale']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        reviews = Order().get_review_score()
        values = self.matching_table[['order_id','seller_id']].merge(reviews, on='order_id', how='inner')
        values['cost_of_bad_reviews'] = values['review_score'].map(lambda x: 100 if x == 1 else (50 if x == 2 else (40 if x == 3 else 0)))
        values = values.groupby('seller_id', as_index=False).agg({'dim_is_five_star': 'mean', \
            'dim_is_one_star': 'mean', 'review_score': 'mean', 'cost_of_bad_reviews': 'sum'})
        values.columns = ['seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score', 'costs']
        return values.dropna()

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        values = self.matching_table[['order_id','seller_id','product_id']]
        df = values.groupby('seller_id', as_index=False).aggregate({'order_id': 'nunique',
                                      'product_id': 'count'})
        df.columns = ['seller_id', 'n_orders', 'quantity']
        df['quantity_per_order'] = values.groupby(['seller_id','order_id'], as_index=False).count().groupby('seller_id', as_index=False).mean()['product_id']
        return df.dropna()

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        orders = self.data['order_items'][['seller_id','price']]
        orders = orders.groupby('seller_id', as_index=False).sum()
        orders['sales'] = orders['price']
        return orders[['seller_id', 'sales']].dropna()

    def get_training_data(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_state', 'seller_city', 'delay_to_carrier',
        'seller_wait_time', 'share_of_five_stars', 'share_of_one_stars',
        'seller_review_score', 'n_orders', 'quantity', 'date_first_sale', 'date_last_sale', 'sales'
        """

        seller_features = self.get_seller_features()
        delay_wait_time = self.get_seller_delay_wait_time()
        active_dates = self.get_active_dates()
        review_score = self.get_review_score()
        quantity = self.get_quantity()
        sales = self.get_sales()
        seller = seller_features.merge(delay_wait_time, on='seller_id', how='inner' )
        seller = seller.merge(active_dates, on='seller_id', how='inner' )
        seller = seller.merge(review_score, on='seller_id', how='inner' )
        seller = seller.merge(quantity, on='seller_id', how='inner' )
        seller = seller.merge(sales, on='seller_id', how='inner' )
        seller['cost_monthly'] = np.floor(((seller['date_last_sale'] - seller['date_first_sale']) / np.timedelta64(1, 'M')))
        seller.loc[seller['cost_monthly'] == 0, 'cost_monthly'] = 1
        seller['cost_monthly']  = seller['cost_monthly'] * 80
        seller['revenues'] = np.round((seller['sales'] * 0.1) + seller['cost_monthly'], 2)
        del seller['cost_monthly']
        seller['profits'] = seller['revenues'] - seller['costs']
        return seller.dropna()

    def get_seller_history(self):
        '''
        Returns a DataFrame aggregating various properties for each product 'category',
        using the aggregation function passed in argument.
        The 'quantity' columns refers to the total number of product sold for this category.
        '''
        orders = self.data['orders'][['order_id', 'order_approved_at']]

        sellers = self.get_training_data()
        order_id = Order().get_training_data()
        sellers = sellers.merge(self.matching_table[['order_id', ]])
        products1 = products.copy().groupby('category').sum()
        products = products.groupby('category').agg(agg)
        products['quantity'] = products1['quantity']
        return products

