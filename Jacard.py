from tqdm import tqdm
import operator
import itertools
from collections import defaultdict


class Jaccard:

    def __init__(self):
        self.item_ratings_sorted = None
        self.train_set = None
        self.item_item_counts = defaultdict(lambda: defaultdict(int))
        self.item_counts = None

    def learn_model(self, train_set):
        self.train_set = train_set
        self.item_counts = self.train_set.groupby('itemID')['userID'].agg('count')

        pbar = tqdm(total=len(train_set.userID.unique()))

        # iterate over the users, and for each user, and each two items that the user has chosen, increase the count
        for u in train_set.userID.unique():
            pbar.update(1)
            userData = self.train_set[self.train_set.userID == u]
            #  For each pair of items in the user data - increase the counts in self.item_item_counts
            for pair in itertools.combinations(userData.itemID.values, 2):
                el1, el2 = pair
                self.item_item_counts[el1][el2] += 1

        pbar.close()

    def get_top_n_recommendations(self, test_set, top_n):
        pbar = tqdm(total=len(test_set.userID.unique()))

        result = {}
        already_ranked_items_by_users = self.train_set.groupby('userID')['itemID'].apply(list)

        # For each user in the test set compute recommendations
        for userID in test_set.userID.unique():
            pbar.update(1)
            # maxvalues will maintain for each potential item to recommend its highest Jaccard score.
            maxvalues = dict()

            # For each such item compute its Jaccard correlation to other items based on the item_item_counts.
            for first in already_ranked_items_by_users[userID]:
                for second in self.item_item_counts[first]:
                    if second not in already_ranked_items_by_users[userID]:
                        maxvalues[second] = (self.item_item_counts[first][second] + self.item_item_counts[second][first]) / (
                                sum(self.item_item_counts[first].values()) + sum(self.item_item_counts[second].values()))

            result[str(userID)] = [i[0] for i in sorted(maxvalues.items(), key=operator.itemgetter(1))[-10:]]

        pbar.close()
        return result

    def clone(self):
        pass


if __name__ == '__main__':
    from util import get_data
    train_set, test_set = get_data()

    jaccard = Jaccard()
    jaccard.learn_model(train_set)
    jaccard_recs = jaccard.get_top_n_recommendations(test_set, top_n=5)

    print(jaccard_recs['431'])
