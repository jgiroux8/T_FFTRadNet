import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch

def RADIal_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []
    class_maps = []

    for radar_FFT,out_label,box_labels,class_map in batch:
        FFTs.append(torch.tensor(radar_FFT).permute(2,0,1))
        # 256,64,32 -> 32,256,64
        # 256,64,256 -> 256,256,64
        encoded_label.append(torch.tensor(out_label))
        labels.append(torch.from_numpy(box_labels))
        class_maps.append(torch.tensor(class_map))

    return torch.stack(FFTs), torch.stack(encoded_label),labels,torch.stack(class_maps)

def CreateDataLoaders(dataset,test_dataset,config=None,seed=0):

    if(config['mode']=='random'):
        n_images = dataset.__len__()

        split = np.array(config['split'])
        if(np.sum(split)!=1.):
            raise NameError('The sum of the train/val split should be equal to 1')
            return

        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        if n_train+n_val != n_images:
            n_val +=1


        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Train Val ratio:', config['split'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset,
                                batch_size=config['train']['batch_size'],
                                shuffle=True,
                                num_workers=config['train']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset,
                                batch_size=config['val']['batch_size'],
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset,
                                batch_size=config['test']['batch_size'],
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        return train_loader,val_loader,test_loader
    elif(config['mode']=='sequence'):
        dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}

        Val_indexes = []
        for seq in Sequences['Validation']:
            idx = np.where(dataset.labels[:,14]==seq)[0]
            Val_indexes.append(dataset.labels[idx,0])
        Val_indexes = np.unique(np.concatenate(Val_indexes))

        Test_indexes = []
        for seq in Sequences['Test']:
            idx = np.where(dataset.labels[:,14]==seq)[0]
            Test_indexes.append(dataset.labels[idx,0])
        Test_indexes = np.unique(np.concatenate(Test_indexes))

        val_ids = [dict_index_to_keys[k] for k in Val_indexes]
        test_ids = [dict_index_to_keys[k] for k in Test_indexes]
        train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Training:', len(train_dataset))
        print('      Validation:', len(val_dataset))
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset,
                                batch_size=config['train']['batch_size'],
                                shuffle=True,
                                num_workers=config['train']['num_workers'],#persistent_workers=True,
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset,
                                batch_size=config['val']['batch_size'],
                                shuffle=False,
                                num_workers=config['val']['num_workers'],#persistent_workers=True,
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset,
                                batch_size=config['test']['batch_size'],
                                shuffle=False,
                                num_workers=config['test']['num_workers'],#persistent_workers=True,
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        return train_loader,val_loader,test_loader

    else:
        raise NameError(config['mode'], 'is not supported !')
        return
