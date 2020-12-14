import numpy as np
import torch

def create_weighted_sampler(dataset, return_weights=False):

    # Check if we have concatenated dataset or just a normal one
    if hasattr(dataset,'datasets'):
        datasets = dataset.datasets
    else:
        datasets = [dataset]

    targets = []

    for subset in datasets:
        # extracting targets from dataset
        temp_targets = subset.get_all_targets()
        # converting to list if it was numpy array
        # more readable format, not ternary operators to convert and append data
        if type(temp_targets)==type(np.empty(1)):
            temp_targets = temp_targets.tolist()

        # appending list
        targets+=temp_targets

    # counting number of sampler in each class
    unique, counts = np.unique(targets, return_counts = True)
    # creating a dict "class":"num of samples"
    counter = dict(zip(unique,counts))

    # after we have counted instances per class, let's calculate total number to calculate further weights
    total = len(targets)

    # now when we know  the total number of occurences we can calculate weight for each class
    class_weights = {}

    for key in counter.keys():
        class_weights[key] = total/float(counter[key])

    # and now let's create proper array with weights assigned to each sample for WeightedRandomSampler from pytorch
    weights = [0]*len(dataset)


    # iterating over each target and assigning its weight
    for i, val in enumerate(targets):
        weights[i] = class_weights[val]

    # if we've set to return only weights, we return it
    if return_weights:
        return weights

    # preparing weights for sampler
    weights = torch.DoubleTensor(weights)
    # creating sampler
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    return sampler

    