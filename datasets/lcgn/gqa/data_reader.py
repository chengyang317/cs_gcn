import threading
import queue
import numpy as np
import json
from tqdm import tqdm
from .. import text_processing
from ..positional_encoding import get_positional_encoding
from .feature_loader import SpatialFeatureLoader, ObjectsFeatureLoader, SceneGraphFeatureLoader
import math
import pt_pack as pt
import torch
from torch.utils.data import IterableDataset


class GqaBatchLoader(object):
    def __init__(self, loader):
        self.loader = loader
        self.work_dir = loader.work_dir
        self.anns = loader.anns
        self.params = params = loader.params
        self.use_filter = params.use_filter
        self.gpu_num = params.gpu_num
        self.batch_size = params.batch_size
        self.vocab_dict = text_processing.VocabDict(self.work_dir.joinpath('vocabulary_gqa.txt'))
        self.text_length = params.text_length
        self.req_field_names = loader.req_field_names
        # peek one example to see whether answer is in the data
        self.has_answer = ('answer' in self.anns[0])
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(self.work_dir.joinpath('answers_gqa.txt'))
        if not self.has_answer:
            print('anns has no answer labels. Using dummy labels.\n\n'
                  '**The final accuracy will be zero (no labels provided)**\n')

        # positional encoding
        self.add_pos_enc = getattr(params, 'add_pos_enc', False)
        self.pos_enc_dim = getattr(params, 'pos_enc_dim', 4)
        assert self.pos_enc_dim % 4 == 0, 'positional encoding dim must be a multiply of 4'
        self.pos_enc_scale = getattr(params, 'pos_enc_scale', 1.)

        self.load_spatial_feat = False
        self.load_object_feat = False
        self.load_scene_graph_feature = False
        feature_type = getattr(params, 'feature_type', 'objects')
        if feature_type == 'spatial':
            self.load_spatial_feat = True
        elif feature_type == 'objects':
            self.load_object_feat = True
        elif feature_type == 'scene_graph':
            self.load_scene_graph_feature = True
        else:
            raise ValueError('Unknown feature type: %s' % feature_type)

        if self.load_spatial_feat:
            spatial_feature_dir = self.work_dir.joinpath('spatial')
            self.spatial_loader = SpatialFeatureLoader(spatial_feature_dir)
            # load one feature map to peek its size
            x = self.spatial_loader.load_feature(self.anns[0]['imageId'])
            self.spatial_D, self.spatial_H, self.spatial_W = x.shape
            # positional encoding
            self.pos_enc = self.pos_enc_scale * get_positional_encoding(
                self.spatial_H, self.spatial_W, self.pos_enc_dim)

        if self.load_object_feat:
            objects_feature_dir = self.work_dir.joinpath('objects')
            self.objects_loader = ObjectsFeatureLoader.build(objects_feature_dir, params.load_mem)
            # load one feature map to peek its size
            self.obj_max_num = getattr(params, 'objects_max_num', 100)
            x, _ = self.objects_loader.load_feature(self.anns[0]['imageId'])
            _, self.obj_dim = x.shape

        if self.load_scene_graph_feature:
            scene_graph_file = self.work_dir.joinpath('train_sceneGraphs.json')
            vocab_name_file = params['vocab_name_file']
            vocab_attr_file = params['vocab_attr_file']
            self.obj_max_num = params.get('objects_max_num', 100)
            self.scene_graph_loader = SceneGraphFeatureLoader(
                scene_graph_file, vocab_name_file, vocab_attr_file,
                max_num=self.obj_max_num)
            # load one feature map to peek its size
            x, _, _ = self.scene_graph_loader.load_feature_normalized_bbox(
                self.anns[0]['imageId'])
            _, self.obj_dim = x.shape

    def load_one_batch(self, sample_ids):
        batch = {}
        node_nums = list()
        actual_b_num = len(sample_ids)
        input_seq_batch = torch.zeros(actual_b_num, self.text_length, dtype=torch.int64)
        seq_length_batch = torch.zeros(actual_b_num, dtype=torch.int64)
        batch['q_labels'] = input_seq_batch
        batch['q_nums'] = seq_length_batch
        if self.load_spatial_feat:
            spatial_feat_batch = np.zeros((actual_b_num, self.spatial_D, self.spatial_H, self.spatial_W),
                                          np.float32)
        if self.load_object_feat or self.load_scene_graph_feature:
            objects_feat_batch = torch.zeros(actual_b_num, self.obj_max_num, self.obj_dim, dtype=torch.float32)
            objects_bbox_batch = torch.zeros(actual_b_num, self.obj_max_num, 4, dtype=torch.float32)
            objects_valid_batch = torch.zeros(actual_b_num, self.obj_max_num, dtype=torch.bool)
            batch['obj_feats'] = objects_feat_batch
            batch['obj_boxes'] = objects_bbox_batch
            batch['obj_masks'] = objects_valid_batch

        qid_list = [None]*actual_b_num
        qstr_list = [None]*actual_b_num
        imageid_list = [None]*actual_b_num
        batch['q_ids'] = qid_list
        if self.has_answer:
            answer_label_batch = torch.zeros(actual_b_num, dtype=torch.int64)
        else:
            answer_label_batch = -torch.ones(actual_b_num, dtype=torch.int64)
        batch['a_labels'] = answer_label_batch
        for n in range(len(sample_ids)):
            iminfo = self.anns[sample_ids[n]]
            question_str = iminfo['question']
            # question_tokens = text_processing.tokenize_gqa(question_str)
            # if len(question_tokens) > self.text_length:
            #     print('data reader: truncating question:\n\t' + question_str)
            #     question_tokens = question_tokens[:self.text_length]
            # question_inds = [self.vocab_dict.word2idx(w) for w in question_tokens]
            q_labels = iminfo['question_labels']
            seq_length = len(q_labels)
            input_seq_batch[n, :seq_length] = torch.tensor(q_labels)
            seq_length_batch[n] = seq_length
            if self.load_spatial_feat:
                feature = self.spatial_loader.load_feature(iminfo['imageId'])
                spatial_feat_batch[n:n+1] = feature
            if self.load_object_feat:
                feature, normalized_bbox, valid = self.objects_loader.load_feature_normalized_bbox(iminfo['imageId'])
                objects_feat_batch[n:n+1] = torch.from_numpy(feature)
                objects_bbox_batch[n:n+1] = torch.from_numpy(normalized_bbox)
                objects_valid_batch[n:n+1] = torch.from_numpy(valid)
                node_nums.append(valid.sum())
            if self.load_scene_graph_feature:
                feature, normalized_bbox, valid = \
                    self.scene_graph_loader.load_feature_normalized_bbox(
                        iminfo['imageId'])
                objects_feat_batch[n:n+1] = feature
                objects_bbox_batch[n:n+1] = normalized_bbox
                objects_valid_batch[n:n+1] = valid
            qid_list[n] = iminfo['question_id']
            qstr_list[n] = question_str
            imageid_list[n] = iminfo['imageId']
            if self.has_answer:
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx

        if self.load_spatial_feat:
            # NCHW -> NHWC
            spatial_feat_batch = spatial_feat_batch.transpose((0, 2, 3, 1))
            batch['spatial_feat_batch'] = spatial_feat_batch
            if self.add_pos_enc:
                # add positional embedding to the image features
                pos_enc_tile = np.tile(
                    self.pos_enc, (len(spatial_feat_batch), 1, 1, 1))
                image_feat_batch = np.concatenate(
                     (spatial_feat_batch, pos_enc_tile), axis=-1)
            else:
                image_feat_batch = spatial_feat_batch
            N, H, W, C = image_feat_batch.shape
            image_feat_batch = image_feat_batch.reshape((N, H*W, C))
            image_valid_batch = np.ones(image_feat_batch.shape[:-1], np.bool)
        if self.load_object_feat or self.load_scene_graph_feature:
            if self.add_pos_enc:
                # add bounding boxes to the object features
                # tile bbox to roughly match the norm of RCNN features
                objects_bbox_tile = self.pos_enc_scale * np.tile(
                    objects_bbox_batch, (1, 1, self.pos_enc_dim//4))
                image_feat_batch = np.concatenate(
                    (objects_feat_batch, objects_bbox_tile), axis=-1)
            else:
                image_feat_batch = objects_feat_batch
        # max_node_num = max(node_nums)
        batch = {key: value for key, value in batch.items() if key in self.req_field_names}

        if self.use_filter:
            batch = self.filter_obj_attr(batch)

        # for key in ('obj_feats', 'obj_boxes', 'obj_masks'):
        #     batch[key] = batch[key][:, :max_node_num]
        return batch

    def filter_obj_attr(self, batch):
        if self.gpu_num is not None:
            gpu_masks = batch['obj_masks'].chunk(self.gpu_num, dim=0)
            node_nums = [mask.sum().item() for mask in gpu_masks]
            feats = torch.cat([batch['obj_feats'], batch['obj_boxes']], dim=-1)
            ret_feats = torch.zeros((self.gpu_num, max(node_nums), feats.shape[-1]), dtype=torch.float32)
            b_feats = feats.chunk(self.gpu_num, dim=0)
            for g_id, b_mask in enumerate(gpu_masks):
                ret_feats[g_id, :node_nums[g_id]] = b_feats[g_id][b_mask]
            batch['img_obj_feats'], batch['img_obj_boxes'] = ret_feats.split((ret_feats.shape[-1]-4, 4), dim=-1)
        return batch

    def __len__(self):
        dset_len = len(self.anns)
        return math.ceil(dset_len / self.batch_size)


class GqaLoader(object):
    def __init__(self, params):
        self.params = params
        self.req_field_names = getattr(params, 'req_field_names',
                                       ('q_labels', 'q_nums', 'obj_feats', 'obj_boxes', 'obj_masks', 'q_ids', 'a_labels')
                                       )
        if getattr(params, 'data_dir', None) is None:
            params.data_dir = 'work_dir/data/gqa_lcgn'
        if getattr(params, 'use_filter', None) is None:
            params.use_filter = True
            params.gpu_num = len(params.gpus.split(','))
        if getattr(params, 'use_thread', None) is None:
            params.use_thread = True
        self.use_thread = params.use_thread
        self.work_dir = pt.to_path(params.data_dir)
        text_length = getattr(params, 'text_length', 30)
        params.text_length = text_length
        anns_json = self.work_dir.joinpath(f'{params.data_split}_anns_{text_length}.json')
        print(f'Loading anns from {anns_json}')
        self.anns = None
        if not anns_json.exists():
            print(f'Creating anns json {anns_json}')
            origin_json = self.work_dir.joinpath(f'questions/{params.data_split}_questions.json')
            vocab_dict = text_processing.VocabDict(self.work_dir.joinpath('vocabulary_gqa.txt'))
            with origin_json.open() as f:
                raw_data = json.load(f)
                qIds = sorted(raw_data)
                for qId, q in tqdm(raw_data.items()):
                    q['question_id'] = qId
                    q_tokens = text_processing.tokenize_gqa(q['question'])[:text_length]
                    q['question_labels'] = [vocab_dict.word2idx(w) for w in q_tokens]
                self.anns = [raw_data[qId] for qId in qIds]
            with anns_json.open('w') as f:
                json.dump(self.anns, f)
        else:
            with anns_json.open('r') as f:
                self.anns = json.load(f)
        print('Done')
        self.shuffle = params.shuffle
        self.prefetch_num = params.prefetch_num
        self.batch_loader = GqaBatchLoader(self)

        if self.use_thread:
        # Start prefetching thread
            self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
            self.prefetch_thread = threading.Thread(
                target=_run_prefetch, args=(self.prefetch_queue, self.batch_loader, self.anns, self.shuffle, self.params)
            )
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()

    def batches(self, one_pass=False):
        if self.use_thread:
            while True:
                # Get a batch from the prefetching queue
                # if self.prefetch_queue.empty():
                #     print('data reader: waiting for IO...')
                batch, n_sample, n_epoch = self.prefetch_queue.get(block=True)
                if batch is None:
                    if one_pass:
                        return
                    else:
                        # get the next batch
                        batch, n_sample, n_epoch = self.prefetch_queue.get(
                            block=True)
                # yield (batch, n_sample, n_epoch)
                return batch
        else:
            return self.batch_loader.load_one_batch()

    def __iter__(self):
        self.iter_batch_ids = iter(range(len(self)))
        return self

    def __next__(self):
        for _ in self.iter_batch_ids:
            return self.batches()

    def __len__(self):
        return len(self.batch_loader)

    @property
    def vocab_num(self):
        return self.batch_loader.vocab_dict.num_vocab

    @property
    def answer_num(self):
        return self.batch_loader.answer_dict.num_vocab


def _run_prefetch(prefetch_queue, batch_loader, anns, shuffle, params):
    num_samples = len(anns)
    batch_size = batch_loader.batch_size

    n_sample = 0
    n_epoch = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put((batch, n_sample, n_epoch), block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            n_sample = 0
            n_epoch += 1
            # Put in a None batch to indicate an epoch is over
            prefetch_queue.put((None, n_sample, n_epoch), block=True)


def load_train_data(params):
    params.data_split = 'train_balanced'
    params.shuffle = True
    params.prefetch_num = 16
    return GqaLoader(params)


def load_eval_data(params):
    params.data_split = 'val_balanced'
    params.shuffle = False
    params.prefetch_num = 16
    return GqaLoader(params)
