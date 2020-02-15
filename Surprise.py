from tqdm import tqdm
import operator
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import PredefinedKFold
from surprise.prediction_algorithms import *


class SurpriseRecMethod:

    # method will be the specific Surprise algorithm that we will call
    def __init__(self, method):
        self.method = method

    def fit(self, train_set):
        self.train_set = train_set

    def get_top_n_recommendations(self, test_set, top_n):
        self.test_set = test_set

        # Surprise requires a slightly different input data format, so we use two different CSVs
        test_path_tmp = "resources//test_file.csv"
        train_path_tmp = "resources//train_file.csv"

        self.train_set.to_csv(train_path_tmp, index=False, header=False)
        self.test_set.to_csv(test_path_tmp, index=False, header=False)

        fold_files = [(train_path_tmp, test_path_tmp)]
        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)

        for trainset, testset in PredefinedKFold().split(data):
            self.method.fit(trainset)

        already_ranked_items_by_users = self.train_set.groupby('userID')['itemID'].apply(list)

        recommendations = {}
        pbar = tqdm(total=len(self.test_set.userID.unique()))
        for userID in self.test_set.userID.unique():
            pbar.update(1)

            if userID not in self.train_set.userID.unique():
                recommendations[str(userID)] = []
                continue

            items_expected_ranking = {}
            for itemID in self.train_set.itemID.unique():
                if itemID in already_ranked_items_by_users[userID]:
                    continue
                # We call here the specific Surprise method that we use for this model
                # The method predicts a score for a given item
                predicted = self.method.predict(str(userID), str(itemID), clip=False)
                items_expected_ranking[itemID] = predicted[3]

            # Now we just sort by decreasing scores and take the top N
            sorted_predictions = sorted(items_expected_ranking.items(), key=operator.itemgetter(1))
            sorted_predictions.reverse()
            sorted_predictions = [x[0] for x in sorted_predictions]
            user_recommendations = sorted_predictions[:top_n]
            recommendations[str(userID)] = user_recommendations
        pbar.close()
        return recommendations


if __name__ == '__main__':
    from util import get_data
    from compare import compute_precision

    train_set, test_set = get_data()
    modelSlopeOne = SurpriseRecMethod(SlopeOne())
    modelSlopeOne.fit(train_set)
    recSlopeOne = modelSlopeOne.get_top_n_recommendations(test_set, 5)
    p3 = compute_precision(test_set, recSlopeOne)
    print(f"SlopeOne: {p3}")

    modelKNNUser = SurpriseRecMethod(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}))
    modelKNNUser.fit(train_set)
    recKNNUser = modelKNNUser.get_top_n_recommendations(test_set, 5)
    p4 = compute_precision(test_set, recKNNUser)

    print(f"KNN: {p4}")
