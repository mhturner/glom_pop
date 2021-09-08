from glom_pop import dataio


path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

included_gloms = dataio.getIncludedGloms(path_to_yaml)

dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)
len(dataset)
dataset


for key in dataset:
    print(key)
    key.split('_')[0]
