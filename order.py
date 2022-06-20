import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''

    def __init__(self, *args):
        # Assign an attribute ".data" to all new instances of Order
        olist = Olist()
        if args:
            self.data = olist.get_data(args[0])
            self.matching_table = olist.get_matching_table(args[0])
        else:
            self.data = Olist().get_data()
            self.matching_table = Olist().get_matching_table()

    def get_wait_time(self, is_delivered=True):
        """
        02-01 > Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filtering out non-delivered orders unless specified
        """
        orders = self.data['orders'].copy()

        if is_delivered == True:
            orders = orders[orders['order_status'] == 'delivered']
        # handle datetime
        colum = ['order_purchase_timestamp',
               'order_approved_at', 'order_delivered_carrier_date',
               'order_delivered_customer_date', 'order_estimated_delivery_date']
        for i in colum:
            orders[i] = pd.to_datetime(orders[i], format='%Y-%m-%d %H:%M:%S')
        # compute wait time
        orders['wait_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600 / 24
        # compute expected wait time
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600 / 24
        # compute delay vs expected - carefully handles "negative" delays
        orders['delay_vs_expected'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.total_seconds() / 3600 / 24
        orders['delay_vs_expected'][orders['order_delivered_customer_date'] < orders['order_estimated_delivery_date']] = 0

        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status', 'order_purchase_timestamp']].dropna()

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        def dim_five_star(x):
            if x == 5:
                return 1
            return 0

        def dim_one_star(x):
            if x == 1:
                return 1
            return 0

        reviews = self.data['order_reviews'].copy()

        reviews["dim_is_five_star"] = reviews["review_score"].map(dim_five_star) # --> Series([0, 1, 1, 0, 0, 1 ...])

        reviews["dim_is_one_star"] = reviews["review_score"].map(dim_one_star) # --> Series([0, 1, 1, 0, 0, 1 ...])
        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]

    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """
        order_items = self.data['order_items'].copy()
        order_items = order_items.groupby('order_id').count()[['order_item_id']].reset_index()
        order_items.columns = ['order_id', 'number_of_products']
        return order_items

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items'].copy()
        order_items = order_items.groupby('order_id')[['seller_id']].nunique().reset_index()
        order_items.columns = ['order_id', 'number_of_sellers']
        return order_items

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items'].copy()[['order_id', 'price','freight_value']]
        return order_items.groupby('order_id').sum().reset_index()

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """
        customer_geolocations = self.data['geolocation'].copy()[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]
        customer_geolocations.columns = ['customer_zip_code_prefix', 'customer_geolocation_lat', 'customer_geolocation_lng']
        customer_geolocations = customer_geolocations.groupby('customer_zip_code_prefix').mean().reset_index()
        seller_geolocations = self.data['geolocation'].copy()[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]
        seller_geolocations.columns = ['seller_zip_code_prefix', 'seller_geolocation_lat', 'seller_geolocation_lng']
        seller_geolocations = seller_geolocations.groupby('seller_zip_code_prefix').mean().reset_index()

        customers = self.data['customers'].copy()[['customer_id', 'customer_zip_code_prefix']]
        sellers = self.data['sellers'].copy()[['seller_id','seller_zip_code_prefix']]

        merged = self.matching_table.merge(customers, on='customer_id', how='inner').merge(sellers, on='seller_id', how='inner')
        merged = merged.merge(customer_geolocations, on='customer_zip_code_prefix', how='inner')
        merged = merged.merge(seller_geolocations, on='seller_zip_code_prefix', how='inner').dropna()
        merged['distance_seller_customer'] = merged.apply(lambda row: haversine_distance(row['customer_geolocation_lng'], row['customer_geolocation_lat'], row['seller_geolocation_lng'], row['seller_geolocation_lat']), axis = 1)
        merged = merged.groupby("order_id", as_index=False).agg(
            {"distance_seller_customer": "mean"}
        )
        return merged[['order_id', 'distance_seller_customer']]

    def get_training_data(self, is_delivered=True,
                          with_distance_seller_customer=False):
        """
        02-01 > Returns a clean DataFrame (without NaN), with the following columns:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status,
        dim_is_five_star, dim_is_one_star, review_score, number_of_products,
        number_of_sellers, freight_value, distance_customer_seller]
        """
        wait = self.get_wait_time(is_delivered)
        sellers = self.get_number_sellers()
        products = self.get_number_products()
        review_score = self.get_review_score()
        price = self.get_price_and_freight()
        merged = wait.merge(sellers, on='order_id', how='inner')
        merged = merged.merge(products, on='order_id', how='inner')
        merged = merged.merge(review_score, on='order_id', how='inner')
        merged = merged.merge(price, on='order_id', how='inner')
        if with_distance_seller_customer == True:
            distance = self.get_distance_seller_customer()
            merged = merged.merge(distance, on='order_id', how='inner')
        return merged
