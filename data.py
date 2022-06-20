import os
import pandas as pd


class Olist:
    FILE_NAMES = ['olist_geolocation_dataset.csv',
     'product_category_name_translation.csv',
     'olist_customers_dataset.csv',
     'olist_sellers_dataset.csv',
     'olist_order_payments_dataset.csv',
     'olist_orders_dataset.csv',
     'olist_order_reviews_dataset.csv',
     'olist_order_items_dataset.csv',
     'olist_products_dataset.csv']

    def get_data(self, *args):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrame loaded from csv files
        """
        csv_path = os.path.join(__file__[0:-7], '../data/csv')
        key_names = self.keys_names()
        data = {}
        if args:
            column = args[0].name
        else:
            column = ''
        for (key_name, file) in zip(key_names, self.FILE_NAMES):
            data[key_name] = pd.read_csv(os.path.join(csv_path, file))
            if column in data[key_name].columns:
                data[key_name] = data[key_name][data[key_name][column].isin(list(args[0]))]
        return data

    def get_matching_table(self, *args):
        """
        01-01 > This function returns a matching table between
        columns [ "order_id", "review_id", "customer_id", "product_id", "seller_id"]
        """
        columns_matching_table = [
            "order_id",
            "review_id",
            "customer_id",
            "product_id",
            "seller_id",
        ]
        if args:
            data = self.get_data(args[0])
        else:
            data = self.get_data()
        frames = []
        for key, df in data.items():
            match_columns = list(set(columns_matching_table) & set(df.columns))
            if key in ['orders','order_reviews', 'order_items']:
                frames.append(data[key][match_columns])
        merged = frames[0].merge(frames[1], on='order_id', how='outer').merge(frames[2], on='order_id', how='outer')
        return merged

    def ping(self):
        """
        You call ping I print pong.
        """
        print('pong')

    def keys_names(self):
        key_names = []
        for i, file in enumerate(self.FILE_NAMES):
            file = file.replace('_dataset.csv', '')
            file = file.replace('.csv', '')
            key_names.append(file.replace('olist_', ''))
        return key_names
