import numpy as np

def compute_precision(test_set, recommendations):
    # hits is the number of items that were recommended and chosen
    hits = 0
    # recs is the total number of recommended items
    recs = 0

    for u in test_set.userID.unique():
        userData = test_set[test_set.userID == u]
        userRecs = recommendations.get(str(u))
        # 5) Compute here the number of hits. Update hits and recs accordingly.
        hits += sum(np.isin(userRecs, list(userData.loc[:, 'itemID'])))
        recs += len(userRecs)

    return hits / recs


if __name__ == '__main__':
    from util import get_data
    from MostPopular import MostPopular
    from Jacard import Jaccard

    train_set, test_set = get_data()
    popular = MostPopular()
    popular.learn_model(train_set)
    popular_recs = popular.get_top_n_recommendations(test_set, top_n=5)

    jaccard = Jaccard()
    jaccard.learn_model(train_set)
    jaccard_recs = jaccard.get_top_n_recommendations(test_set, top_n=5)

    p1 = compute_precision(test_set, jaccard_recs)
    p2 = compute_precision(test_set, popular_recs)
    print("Jaccard=", p1, "  Popularity=", p2)