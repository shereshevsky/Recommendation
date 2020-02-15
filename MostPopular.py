class MostPopular:

    def __init__(self):
        self.item_ratings_sorted = None
        self.train_set = None

    def learn_model(self, train_set):
        self.train_set = train_set
        # 1) Add code to set the item_ratings_sorted to the list of items in the training set,
        # ordered by decreasing popularity (i.e., the number of users who have chosen an item)
        self.item_ratings_sorted = list(train_set.groupby("itemID").count().sort_values(by='userID', ascending=False).index)

    def get_top_n_recommendations(self, test_set, top_n):
        result = {}
        already_ranked_items_by_user = self.train_set.groupby('userID')['itemID'].apply(list)

        # For each user in the test set compute recommendations
        for userID in test_set.userID.unique():
            result[str(userID)] = [i for i in self.item_ratings_sorted[:top_n+len(already_ranked_items_by_user[userID])]
                                   if i not in already_ranked_items_by_user[userID]][:top_n]
        return result

    def clone(self):
        pass


if __name__ == '__main__':
    from util import get_data
    from compare import compute_precision

    train_set, test_set = get_data()
    mp = MostPopular()
    mp.learn_model(train_set)

    recommendations = mp.get_top_n_recommendations(test_set, 10)

    popular = MostPopular()
    popular.learn_model(train_set)
    popular_recs = popular.get_top_n_recommendations(test_set, top_n=5)

    print(compute_precision(test_set, popular_recs))
