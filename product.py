import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order



class Product:

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

    def get_product_features(self):
        """
        Returns a DataFrame with:
       'product_id', 'product_category_name', 'product_name_length',
       'product_description_length', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm'
        """

        products = self.data['products']

        en_category = self.data['product_category_name_translation']
        df = products.merge(en_category, on='product_category_name')
        df.drop(['product_category_name'], axis=1, inplace=True)
        df.rename(columns={'product_category_name_english': 'category',
                  'product_name_lenght': 'product_name_length',
                  'product_description_lenght': 'product_description_length'},
                  inplace=True)


        return df

    def get_price(self):
        """
        Return a DataFrame with:
        'product_id', 'price'
        """
        order_items = self.data['order_items']

        return order_items[['product_id', 'price']].groupby('product_id').mean()

    def get_wait_time(self):
        """
        Returns a DataFrame with:
        'product_id', 'wait_time'
        """
        matching_table = self.matching_table
        orders_wait_time = self.order.get_wait_time()

        df = matching_table.merge(orders_wait_time,
                                 on='order_id')

        return df.groupby('product_id',
                          as_index=False).agg({'wait_time': 'mean'})

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        matching_table = self.matching_table
        orders_reviews = self.order.get_review_score()


        matching_table = matching_table[['order_id',
                                         'product_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews,
                                  on='order_id')
        df['cost_of_bad_reviews'] = df['review_score'].map(lambda x: 100 if x == 1 else (50 if x == 2 else (40 if x == 3 else 0)))

        df = df.groupby('product_id',
                        as_index=False).agg({'dim_is_one_star': 'mean',
                                             'dim_is_five_star': 'mean',
                                             'review_score': 'mean',
                                             'cost_of_bad_reviews': 'sum'
                                             })
        df.columns = ['product_id', 'share_of_one_stars',
                      'share_of_five_stars', 'review_score', 'cost']
        return df


    def get_quantity(self):
        """
        Returns a DataFrame with:
        'product_id', 'n_orders', 'quantity'
        """
        order_items = self.data['order_items']

        n_orders =\
            order_items.groupby('product_id')['order_id'].nunique().reset_index()
        n_orders.columns = ['product_id', 'n_orders']

        quantity = \
            order_items.groupby('product_id',
                                   as_index=False).agg({'order_id': 'count'})
        quantity.columns = ['product_id', 'quantity']

        return n_orders.merge(quantity, on='product_id')

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['product_id', 'price']]\
            .groupby('product_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_training_data(self):

        training_set =\
            self.get_product_features()\
                .merge(
                self.get_wait_time(), on='product_id'
               ).merge(
                self.get_price(), on='product_id'
               ).merge(
                self.get_review_score(), on='product_id'
               ).merge(
                self.get_quantity(), on='product_id'
               ).merge(
                self.get_sales(), on='product_id'
               )

        olist_sales_cut = 0.1
        training_set['revenues'] = olist_sales_cut * training_set['sales']
        training_set['profits'] = training_set['revenues'] - training_set['cost']
        return training_set

    def get_product_cat(self, agg="median"):
        '''
        Returns a DataFrame aggregating various properties for each product 'category',
        using the aggregation function passed in argument.
        The 'quantity' columns refers to the total number of product sold for this category.
        '''
        products = self.get_training_data()
        products1 = products.copy().groupby('category').sum()
        products = products.groupby('category').agg(agg)
        products['quantity'] = products1['quantity']
        return products
