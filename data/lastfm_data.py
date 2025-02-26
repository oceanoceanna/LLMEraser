import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random

class LastfmData(data.Dataset):
    def __init__(self, seed =1234, ratio = 0.05,mode_type = 'normal',data_dir=r'data/ref/lastfm',
                 stage='train',
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.ratio = ratio
        self.seed = seed
        self.mode_type = mode_type
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=4606
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_music_id2name()
        if self.mode_type == 'normal':
            if self.stage=='train':
                filename="train_data.df"
            elif self.stage=='val':
                filename="Val_data.df"
            elif self.stage=='test':
                filename="Test_data.df"
            data_path=op.join(self.data_dir, filename)
            self.session_data = self.session_data4frame(data_path, self.item_id2name) 
        elif self.mode_type == 'delete':
            formatted_ratio = "{:.2f}".format(self.ratio)
            if self.stage=='all':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/train_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='before':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/before_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='after':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/after_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='val':
                filename="Val_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)
            elif self.stage=='test':
                filename="Test_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)
        elif self.mode_type == 'attack':
            formatted_ratio = "{:.2f}".format(self.ratio)
            if self.stage=='all':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/train_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='before':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/before_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='after':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/after_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='val':
                filename="Val_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)
            elif self.stage=='test':
                filename="Test_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)
        elif self.mode_type == 'attack_label':
            formatted_ratio = "{:.2f}".format(self.ratio)
            if self.stage=='all':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/train_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='before':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/before_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='after':
                filename="/data3/dingcl/LLaRA/data/ref/lastfm/"+str(self.mode_type)+"/after_data_"+str(self.mode_type) +'_'+ formatted_ratio + '.df'
                self.session_data = pd.read_pickle(filename)
            elif self.stage=='val':
                filename="Val_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)
            elif self.stage=='test':
                filename="Test_data.df"
                data_path=op.join(self.data_dir, filename)
                self.session_data = self.session_data4frame(data_path, self.item_id2name)


        
    def get_music_id2name(self):
        music_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                music_id2name[int(ll[0])] = ll[1].strip()
        return music_id2name
    
    def session_data4frame(self, datapath, music_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        def seq_to_title(x): 
            return [music_id2name[x_i] for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return music_id2name[x]
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        return train_data


class LastfmData_attack(data.Dataset):
    def __init__(self, seed =1234, ratio = 0.05,mode_type = 'normal',data_dir=r'data/ref/lastfm',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.ratio = ratio
        self.seed = seed
        self.mode_type = mode_type
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=4606
        self.item_id2name=self.get_music_id2name()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

        
    def get_music_id2name(self):
        music_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                music_id2name[int(ll[0])] = ll[1].strip()
        return music_id2name
    
    def session_data4frame(self, datapath, music_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        def seq_to_title(x): 
            return [music_id2name[x_i] for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return music_id2name[x]
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        return train_data
    
    def get_attack_data(self):
        filename="train_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)
        set1,set2,set3 = self.get_train_data_after_attack()
        return set1,set2,set3
    
    def get_attack_label_data(self):
        filename="train_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)
        set1,set2,set3 = self.get_train_data_after_attack_label()
        return set1,set2,set3
    
    def get_delete_data(self):
        filename="train_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)
        set1,set2,set3 = self.get_train_data_after_delete()
        return set1,set2,set3
    
    def attack_label_row(self,row):
        candidates = [i for i in list(self.item_id2name.keys()) if i not in row['seq_unpad'] and i != row['next']]
        noisy_item = random.choice(candidates)
        noist_item_title = self.item_id2name[noisy_item]

        row['next'] = noisy_item
        row['next_item_name'] = noist_item_title
        return row
    
    def delete_row(self,row):
        if row['len_seq'] > 3:
            last_valid_index = row['len_seq'] - 1
            new_seq = row['seq'][:last_valid_index] + [self.padding_item_id] * (len(row['seq']) - len(row['seq'][:last_valid_index]))
            row['seq'] = new_seq
            row['seq_title'] = row['seq_title'][:-1]
            row['len_seq'] = row['len_seq'] - 1
            row['seq_unpad'] = row['seq_unpad'][:-1]
        return row
    
    def attack_row(self,row):
        candidates = [i for i in list(self.item_id2name.keys()) if i not in row['seq_unpad'] and i != row['next']]
        noisy_item = random.choice(candidates)
        noist_item_title = self.item_id2name[noisy_item]
        last_valid_index = row['len_seq'] - 1
        if len(row['seq'])<10:
            row['seq'] = row['seq'][:last_valid_index + 1] + [noisy_item] + [self.padding_item_id] * (len(row['seq']) - last_valid_index - 2)
            row['seq_unpad'].append(noisy_item)
            row['seq_title'] = row['seq_title'] + [noist_item_title]
            row['len_seq'] += 1
        else: 
            row['seq'] = row['seq'][:-1] + [noisy_item] 
            row['seq_unpad'] = row['seq_unpad'][:-1] + [noisy_item] 
            row['seq_title'] = row['seq_title'][:-1] + [noist_item_title]
        return row
    
    def get_train_data_after_delete(self):
        filtered_df = self.session_data[self.session_data['len_seq'] > 3]
        total_rows = len(self.session_data)
        num_samples_to_remove = int(self.ratio * total_rows)

        samples_to_remove = filtered_df.sample(n=num_samples_to_remove, random_state=self.seed)
        processed_samples = samples_to_remove.apply(self.delete_row, axis=1)
        assert processed_samples.shape[0] == num_samples_to_remove

        remaining_data = self.session_data[~self.session_data.index.isin(samples_to_remove.index)]
        final_df = pd.concat([processed_samples,remaining_data])
        assert len(final_df) == len(self.session_data)
        return samples_to_remove,processed_samples,final_df
    
    def get_train_data_after_attack(self):
        filtered_df = self.session_data
        print(filtered_df.shape)
        total_rows = len(self.session_data)
        num_samples_to_modify = int(self.ratio * total_rows)
        print(num_samples_to_modify)
        samples_to_modify = filtered_df.sample(n=num_samples_to_modify, random_state=self.seed)


        processed_samples = samples_to_modify.apply(self.attack_row, axis=1)


        assert processed_samples.shape[0] == num_samples_to_modify
        remaining_data = self.session_data[~self.session_data.index.isin(samples_to_modify.index)]
        result_df = pd.concat([processed_samples,remaining_data])
        assert len(result_df) == len(self.session_data)
        return samples_to_modify,processed_samples,result_df
    
    def get_train_data_after_attack_label(self):
        filtered_df = self.session_data
        total_rows = len(self.session_data)
        num_samples_to_modify = int(self.ratio * total_rows)
        samples_to_modify = filtered_df.sample(n=num_samples_to_modify, random_state=self.seed)

        processed_samples = samples_to_modify.apply(self.attack_label_row, axis=1)

        assert processed_samples.shape[0] == num_samples_to_modify
        remaining_data = self.session_data[~self.session_data.index.isin(samples_to_modify.index)]
        result_df = pd.concat([processed_samples,remaining_data])
        assert len(result_df) == len(self.session_data)
        return samples_to_modify,processed_samples,result_df