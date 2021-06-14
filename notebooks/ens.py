import argparse
import math
import os
from heapq import heappush, heappop, heappushpop

import matplotlib.pyplot as plotter
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enspath", type=str, default="./data", help="Path to folder with all csvs")
    parser.add_argument("--enstype", type=str, default="loop",
                        help="Type of ensembling to be performed - Current options: loop / sa")
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    parser.add_argument('--subdata', action='store_const', default=False, const=True)

    # Parse the arguments.
    args = parser.parse_args()

    return args

### HELPERS ###

# Acc = (TP + TN)/(TP + TN + FP + FN) = (TP + TN) / P + N   (= Correct ones / all)
# Senstivity / tpr = TP / P 
# Specificity / tnr = TN / N


def acc_from_roc(labels, probas, splits=None):
    '''Determines the greatest achievable accuracy from the ROC curve.'''
    if splits is None:
        splits = (250, 250)

    fpr, tpr, thresholds = roc_curve(labels, probas)
    tp = tpr * splits[0]
    tn = (1 - fpr) * splits[1]
    acc = (tp + tn) / np.sum(splits)
    best_threshold = thresholds[np.argmax(acc)]

    return np.amax(acc), best_threshold


def average(data, weights=None):
    N = data.shape[1]
    if weights is None:
        weights = [1 / N] * N
    elif np.sum(weights) != 1.:
        weights = weights / np.sum(weights)

    # Compute weighted avg
    return data.apply(lambda row: row.multiply(weights).sum(), axis=1)

### SIMPLEX ###

### Similar to scipy optimize
# Taken & adapted from:
# https://github.com/chrisstroemel/Simple

CAPACITY_INCREMENT = 1000


class _Simplex:
    def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction,
                 difference):
        self.pointIndices = pointIndices
        self.testCoords = testCoords
        self.contentFractions = contentFractions
        self.contentFraction = contentFraction
        self.__objectiveScore = objectiveScore
        self.__opportunityCost = opportunityCost
        self.update(difference)

    def update(self, difference):
        self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
        self.difference = difference

    def __eq__(self, other):
        return self.acquisitionValue == other.acquisitionValue

    def __lt__(self, other):
        return self.acquisitionValue < other.acquisitionValue


class SimpleTuner:
    def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
        self.__cornerPoints = cornerPoints
        self.__numberOfVertices = len(cornerPoints)
        self.queue = []
        self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
        self.testPoints = np.empty((self.capacity, self.__numberOfVertices))
        self.objective = objectiveFunction
        self.iterations = 0
        self.maxValue = None
        self.minValue = None
        self.bestCoords = []
        self.opportunityCostFactor = exploration_preference  # / self.__numberOfVertices

    def optimize(self, maxSteps=10):
        for step in range(maxSteps):
            # print(self.maxValue, self.iterations, self.bestCoords)
            if len(self.queue) > 0:
                targetSimplex = self.__getNextSimplex()
                newPointIndex = self.__testCoords(targetSimplex.testCoords)
                for i in range(0, self.__numberOfVertices):
                    tempIndex = targetSimplex.pointIndices[i]
                    targetSimplex.pointIndices[i] = newPointIndex
                    newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
                    newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
                    heappush(self.queue, newSimplex)
                    targetSimplex.pointIndices[i] = tempIndex
            else:
                testPoint = self.__cornerPoints[self.iterations]
                testPoint.append(0)
                testPoint = np.array(testPoint, dtype=np.float64)
                self.__testCoords(testPoint)
                if self.iterations == (self.__numberOfVertices - 1):
                    initialSimplex = self.__makeSimplex(np.arange(self.__numberOfVertices, dtype=np.intp), 1)
                    heappush(self.queue, initialSimplex)
            self.iterations += 1

    def get_best(self):
        return (self.maxValue, self.bestCoords[0:-1])

    def __getNextSimplex(self):
        targetSimplex = heappop(self.queue)
        currentDifference = self.maxValue - self.minValue
        while currentDifference > targetSimplex.difference:
            targetSimplex.update(currentDifference)
            # if greater than because heapq is in ascending order
            if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
                targetSimplex = heappushpop(self.queue, targetSimplex)
        return targetSimplex

    def __testCoords(self, testCoords):
        objectiveValue = self.objective(testCoords[0:-1])
        if self.maxValue == None or objectiveValue > self.maxValue:
            self.maxValue = objectiveValue
            self.bestCoords = testCoords
            if self.minValue == None: self.minValue = objectiveValue
        elif objectiveValue < self.minValue:
            self.minValue = objectiveValue
        testCoords[-1] = objectiveValue
        if self.capacity == self.iterations:
            self.capacity += CAPACITY_INCREMENT
            self.testPoints.resize((self.capacity, self.__numberOfVertices))
        newPointIndex = self.iterations
        self.testPoints[newPointIndex] = testCoords
        return newPointIndex

    def __makeSimplex(self, pointIndices, contentFraction):
        vertexMatrix = self.testPoints[pointIndices]
        coordMatrix = vertexMatrix[:, 0:-1]
        barycenterLocation = np.sum(vertexMatrix, axis=0) / self.__numberOfVertices

        differences = coordMatrix - barycenterLocation[0:-1]
        distances = np.sqrt(np.sum(differences * differences, axis=1))
        totalDistance = np.sum(distances)
        barycentricTestCoords = distances / totalDistance

        euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)

        vertexValues = vertexMatrix[:, -1]

        testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
        testPointDistances = np.sqrt(np.sum(testpointDifferences * testpointDifferences, axis=1))

        inverseDistances = 1 / testPointDistances
        inverseSum = np.sum(inverseDistances)
        interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum

        currentDifference = self.maxValue - self.minValue
        opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

        return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue,
                        opportunityCost, contentFraction, currentDifference)

    def plot(self):
        if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
        matrix = self.testPoints[0:self.iterations, :]

        x = matrix[:, 0].flat
        y = matrix[:, 1].flat
        z = matrix[:, 2].flat

        coords = []
        acquisitions = []

        for triangle in self.queue:
            coords.append(triangle.pointIndices)
            acquisitions.append(-1 * triangle.acquisitionValue)

        plotter.figure()
        plotter.tricontourf(x, y, coords, z)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()

        plotter.figure()
        plotter.tripcolor(x, y, coords, acquisitions)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()

        plotter.show()


def Simplex(devs, label, df_list=False, exploration=0.01, scale=1):
    """
    devs: list of dataframes with "proba" column
    label: list/np array of ground truths
    scale: By default we will get weights in the 0-1 range. Setting e.g. scale=50, gives weights in the 0-50 range.
    """
    predictions = []
    if df_list:
        for df in devs:
            predictions.append(df.proba)

        print(len(predictions[0]))
    else:
        for i, column in enumerate(devs):
            predictions.append(devs.iloc[:, i])

        print(len(predictions[0]))

    print("Optimizing {} inputs.".format(len(predictions)))

    def roc_auc(weights):
        ''' Will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return roc_auc_score(label, final_prediction)

    # This defines the search area, and other optimization parameters.
    # For e.g. 11 models, we have 12 corner points -- e.g. all none, only model 1, all others none, only model 2 all others none..
    # We concat an identity matrix & a zero array to create those
    zero_vtx = np.zeros((1, len(predictions)), dtype=int)
    optimization_domain_vertices = np.identity(len(predictions), dtype=int) * scale

    optimization_domain_vertices = np.concatenate((zero_vtx, optimization_domain_vertices), axis=0).tolist()

    number_of_iterations = 3000
    exploration = exploration  # optional, default 0.01

    # Optimize weights
    tuner = SimpleTuner(optimization_domain_vertices, roc_auc, exploration_preference=exploration)
    tuner.optimize(number_of_iterations)
    best_objective_value, best_weights = tuner.get_best()

    print('Optimized =', best_objective_value)  # same as roc_auc(best_weights)
    print('Weights =', best_weights)

    return best_weights


### APPLYING THE HELPER FUNCTIONS ###

def sa_wrapper(data_path="./data"):
    """
    Applies simple average.

    data_path: path to folder with  X * (dev_seen, test_seen & test_unseen) .csv files
    """
    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]
    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {}  # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            print("Included in Simple Average: ", csv)
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values

    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    dev_SA = simple_average(dev_probas, dev[0])
    test_SA = simple_average(test_probas, test[0])
    test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0])

    # Create output dir
    os.makedirs(os.path.join(data_path, args.exp), exist_ok=True)

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                os.remove(os.path.join(data_path, csv))
                dev_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_dev_seen_SA.csv"), index=False)
            elif "test_unseen" in csv:
                os.remove(os.path.join(data_path, csv))
                test_unseen_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_unseen_SA.csv"), index=False)
            elif "test" in csv:
                os.remove(os.path.join(data_path, csv))
                test_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_seen_SA.csv"), index=False)


def main(path, gt_path="./data/"):
    """
    Loops through Averaging, Power Averaging, Rank Averaging, Optimization to find the best ensemble.

    path: String to directory with csvs of all models
    For each model there should be three csvs: dev, test, test_unseen

    gt_path: Path to folder with ground truth for dev
    """
    # Ground truth
    gt = pd.read_json(os.path.join(gt_path, 'dev_seen.jsonl'), lines=True)

    dev, ts, tu = {}, {}, {}
    print('Loading data:')
    for csv in sorted(os.listdir(path)):
        if ".csv" in csv:
            print(csv)
            name = csv.split('_')[0]
            if ("dev" in csv) or ("val" in csv):
                dev[name] = pd.read_csv(os.path.join(path, csv))
                dev_idx = dev[name].id.values
            elif "test_unseen" in csv:
                tu[name] = pd.read_csv(os.path.join(path, csv))
                tu_idx = tu[name].id.values
            elif "test_seen" in csv:
                ts[name] = pd.read_csv(os.path.join(path, csv))
                ts_idx = ts[name].id.values

    dev_probas = pd.DataFrame({k: v.proba.values for k, v in dev.items()})
    # dev_probas.set_index(dev_idx, inplace=True)
    ts_probas = pd.DataFrame({k: v.proba.values for k, v in ts.items()})
    # ts_probas.set_index(ts_idx, inplace=True)
    tu_probas = pd.DataFrame({k: v.proba.values for k, v in tu.items()})
    # tu_probas.set_index(tu_idx, inplace=True)

    # TODO
    '''
    if len(dev_df) > len(dev_probas):
        #print("Your predictions do not include the full dev!")
        #dev_df = dev[0][["id"]].merge(dev_df, how="left", on="id")
    '''

    loop, last_score, delta = 0, 0, 0.1

    while delta > 0.0001:

        # Individual AUROCs
        print('\n' + '-' * 21, 'ROUND ' + str(loop), '-' * 21)
        print("Individual AUROCs for Validation Sets:\n")
        for i, column in enumerate(dev_probas):
            score = roc_auc_score(gt.label, dev_probas.iloc[:, i])
            print(column, score)

        # Drop worst performing sets
        if loop > 0:
            print('\n' + '-' * 50)
            scores = dev_probas.apply(lambda col: roc_auc_score(gt.label, col), result_type='reduce')
            while len(scores) > 5:
                worst = scores.idxmin()
                # del dev[worst]
                dev_probas.drop(worst, axis=1, inplace=True)
                ts_probas.drop(worst, axis=1, inplace=True)
                tu_probas.drop(worst, axis=1, inplace=True)
                scores.drop(worst, inplace=True)
                print("Dropped:", worst)

        # Spearman Correlations:
        print('\n' + '-' * 50)
        print("Spearman Corrs:")
        dev_corr = dev_probas.corr(method='spearman')
        test_seen_corr = ts_probas.corr(method='spearman')
        test_unseen_corr = tu_probas.corr(method='spearman')

        print('\n', dev_corr)
        print('\n', test_seen_corr)
        print('\n', test_unseen_corr)
        print('\n' + '-' * 50)

        # Simple
        print('Simple:')
        weights_dev = Simplex(dev_probas, gt.label)
        dev_probas[f'dev_SX_{loop}'] = average(dev_probas, weights=weights_dev)
        ts_probas[f'ts_SX_{loop}'] = average(ts_probas, weights=weights_dev)
        tu_probas[f'tu_SX_{loop}'] = average(tu_probas, weights=weights_dev)
        score = roc_auc_score(gt.label, dev_probas[f'dev_SX_{loop}'])
        print(f"AUROC: {score:.4f}")
        print(f"Accuracy: {acc_from_roc(gt.label, dev_probas[f'dev_SX_{loop}'])[0]:.4f}")
        print('\n' + '-' * 50)

        # Arithmetic Mean
        print('Arithmetic Mean:')
        dev_probas[f'dev_AM_{loop}'] = average(dev_probas.apply(np.exp))
        ts_probas[f'ts_AM_{loop}'] = average(ts_probas.apply(np.exp))
        tu_probas[f'tu_AM_{loop}'] = average(tu_probas.apply(np.exp))
        print(f"AUROC: {roc_auc_score(gt.label, dev_probas[f'dev_AM_{loop}']):.4f}")
        print(f"Accuracy: {acc_from_roc(gt.label, dev_probas[f'dev_AM_{loop}'])[0]:.4f}")
        print('\n' + '-' * 50)

        # Geometric Mean (remain in logspace)
        print('Geometric Mean:')
        dev_probas[f'dev_GM_{loop}'] = average(dev_probas)
        ts_probas[f'ts_GM_{loop}'] = average(ts_probas)
        tu_probas[f'tu_GM_{loop}'] = average(tu_probas)
        print(f"AUROC: {roc_auc_score(gt.label, dev_probas[f'dev_GM_{loop}']):.4f}")
        print(f"Accuracy: {acc_from_roc(gt.label, dev_probas[f'dev_GM_{loop}'])[0]:.4f}")
        print('\n' + '-' * 50)

        # TODO: Power Average
        '''
        print('Power Average:')
        dev_PA = simple_average(dev_probas, dev[0], power=2, normalize=True)
        test_PA = simple_average(test_probas, test[0], power=2, normalize=True)
        test_unseen_PA = simple_average(test_unseen_probas, test_unseen[0], power=2, normalize=True)
        print(roc_auc_score(dev_df.label, dev_PA.proba), accuracy_score(dev_df.label, dev_PA.label))
        print('\n' + '-' * 50)
        '''

        # Rank Average
        print('Rank Average:')
        dev_probas[f'dev_RA_{loop}'] = average(dev_probas.apply(lambda col: rankdata(col) / len(col)))
        ts_probas[f'ts_RA_{loop}'] = average(ts_probas.apply(lambda col: rankdata(col) / len(col)))
        tu_probas[f'tu_RA_{loop}'] = average(tu_probas.apply(lambda col: rankdata(col) / len(col)))
        print(f"AUROC: {roc_auc_score(gt.label, dev_probas[f'dev_RA_{loop}']):.4f}")
        print(f"Accuracy: {acc_from_roc(gt.label, dev_probas[f'dev_RA_{loop}'])[0]:.4f}")
        print('\n' + '-' * 50)

        # Calculate Delta & increment loop
        delta = abs(score - last_score)
        last_score = score

        loop += 1

        print("Currently at {} after {} loops.".format(last_score, loop))

    dev_best = dev_probas[f'dev_SX_{loop - 1}']
    ts_best = ts_probas[f'ts_SX_{loop - 1}']
    tu_best = tu_probas[f'tu_SX_{loop - 1}']

    # Get accuracy thresholds & optimize (This does not add value to the roc auc, but just to also have an acc score)
    acc, threshold = acc_from_roc(gt.label, dev_best)

    # As Simplex at some point simply weighs the highest of all - lets take sx as the final prediction after x loops
    ts_labels = ts_best.apply(lambda x: 1 if x > threshold else 0)
    ts_out = pd.DataFrame({'id': ts_idx, 'proba': ts_best, 'label': ts_labels})
    tu_labels = tu_best.apply(lambda x: 1 if x > threshold else 0)
    tu_out = pd.DataFrame({'id': tu_idx, 'proba': tu_best, 'label': tu_labels})
    ts_out.to_csv(os.path.join(path, f"final/FIN_test_seen_{args.exp}_{loop}.csv"), index=False)
    tu_out.to_csv(os.path.join(path, f"final/FIN_test_unseen_{args.exp}_{loop}.csv"), index=False)

    print("Finished.")


if __name__ == "__main__":

    args = parse_args()

    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")
