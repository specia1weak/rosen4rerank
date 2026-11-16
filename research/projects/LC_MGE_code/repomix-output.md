This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
- Pay special attention to the Repository Description. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: README.md, CORE_CODE.md, repomix.config.json, rerank_model_dir/MultiG_E_Submodularity_Model_lc/config.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/data_input.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/layers.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/main.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/model.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/tools.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/util.py, rerank_model_dir/MultiG_E_Submodularity_Model_lc/run.sh, rerank_model_dir/MultiG_E_model/config.py, rerank_model_dir/MultiG_E_model/model.py, rerank_model_dir/MultiG_E_Submodularity_Model/config.py, rerank_model_dir/MultiG_E_Submodularity_Model/model.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/.DS_Store, **/avg_std/**, **/model/**, **/log*, **/*.zip, rerank_model_dir/rm.md, repomix-output.*
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style

# User Provided Header
LC_MGE core code. Start with CORE_CODE.md. The LC implementation includes its training/input/model dependencies; the two related versions include model.py and config.py for comparison only. Code bodies, comments, and interior blank lines are preserved; Repomix normalizes boundary whitespace.

# Directory Structure
````
rerank_model_dir/
  MultiG_E_model/
    config.py
    model.py
  MultiG_E_Submodularity_Model/
    config.py
    model.py
  MultiG_E_Submodularity_Model_lc/
    config.py
    data_input.py
    layers.py
    main.py
    model.py
    run.sh
    tools.py
    util.py
CORE_CODE.md
README.md
repomix.config.json
````

# Files

## File: rerank_model_dir/MultiG_E_model/config.py
````python
# coding: utf-8 -*-
import shutil
import time
import os

# result
RANDOM_SEED = 2022
BATCH_SIZE = 1024
IMP_LOSS_WEIGHT = 0.02
# basic config
EPOCH = 5 
LEARNING_RATE = 0.001
DATA_MODE = 3 # 1:local train，2:local test, 3:docker evaluate
TRAIN_MODE = 3
MODEL_NAME = "pier_listwise_whole_v4"

USE_CONSTRATIVE_LOSS = True
CONSTRATIVE_LOSS_K = 0.01

# poi类别特征
FEATURE_CATE_NUM = 7 # v1r3:19
# dense特征
FEATURE_DENSE_NUM = 5  # v1:28 v1r2:79 v1r3:83
# 预估值特征
FEATURE_CXR_NUM = 3
# 环境特征
FEATURE_ENV_NUM = 2
# 自然poi
FEATURE_NATURE_POI = 25

# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# hash vector
HASH_VECTOR = [[0.4011729380078948, 0.9907933488085712, 0.889170658914898, 0.015553701343792192, 0.2269008585804, 0.889609750280199, 0.19280345796299014, 0.33362195188257815],
               [0.1174580997476552, 0.4475570392889796, 0.9925939893135071, 0.7296640075216434, 0.6436313332140967, 0.8568068597644793, 0.9018604021483339, 0.28774343258170776],
               [0.4374993384759095, 0.6807149381274915, 0.20502126763268802, 0.6968687323602859, 0.6449029002244834, 0.08732080642447282, 0.6119476780855001, 0.8616788453789646],
               [0.19344266090314144, 0.4268233179146762, 0.10951597767480326, 0.9867838283258178, 0.8340011944969644, 0.7992329879482085, 0.005303560724105649, 0.9662924610057512],
               [0.5824282763301396, 0.5090708710943849, 0.4462703076663568, 0.5482943153972023, 0.06782829736851825, 0.08907408658464577, 0.13400486343251583, 0.1848228429919272]]

POSITION_ENCODING = [[0.08790239841717873, 0.033267486152506076, 0.5495130189114207, 0.17419777583517537, 0.6838981992197484, 0.07935154925635501, 0.02665372302227631, 0.4181008411574786],
                     [0.02038159041970189, 0.9310485215006198, 0.723661313625571, 0.5110323516977285, 0.7812851438212606, 0.40722710713947474, 0.04646490014890503, 0.9565353323381218],
                     [0.17279361698259843, 0.12968275664201512, 0.3183103529758954, 0.6360591081256931, 0.1558507653689548, 0.5972802646455662, 0.4380619835390329, 0.6088094249662641],
                     [0.038262298606345335, 0.9999545626217287, 0.7113776275017341, 0.5434618368150265, 0.6853527957705402, 0.9662653254145415, 0.9641592716989676, 0.3443813983264],
                     [0.7903243938847678, 0.9952713339078417, 0.8741415264071601, 0.45665348276461737, 0.7693872696125916, 0.772509599868299, 0.2540369924156157, 0.24781240400239857]]


TIME_AWARE_WEIGHT = [[1/2,1/2,1/3,1/3,1/4]]
PERMUATION_SIZE = 120
TOP_K = 20


EXPOSE_RATE_FOR_BEAM_SEARCH = [[1.0,0.9,0.8,0.7,0.6]]

# N: Cut Number of POI For Train
POI_NUM = 5
FEATURE_NUM = 9
PAGE_NUM = 5
FEATURE_NUM_FOR_PAGE = 11

# 属性特征：KA AOR BRAND
FEATURE_ATTR_NUM = 3

# DELIVERY_FEAT
DELIVERY_FEAT_NUM = 4

# OUT NUM
OUT_NUM = 1

PLACE_HOLDER_NUM = 11
DENSE_FEAT_NUM = 439


# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 20],
    'INPUT_TENSOR_LAYERS_B': [128, 32],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
# A_INPUT_DIM = POI_NUM * (MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)
MLP_INPUT_DIM = CATE_FEATURE_EMBEDDINGS_SHAPE[1] * 3 + 1 + MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1]
DIN_CONF = {}

# train data
# /users/lemonace/Downloads/tfrecord-rl-limit5-v1
if DATA_MODE == 1:
    TRAIN_FILE = ['/users/meituan_sxw/Downloads/part-r-00046']
    VALID_FILE = TRAIN_FILE
    PREDICT_FILE = VALID_FILE
    TEST_FILE = PREDICT_FILE
elif DATA_MODE == 2:
    TRAIN_FILE = ['/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/test_data/part-r-*']
    VALID_FILE = TRAIN_FILE
    TEST_FILE = VALID_FILE
elif DATA_MODE == 3:
    TRAIN_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/train_data/part-r-000*1"]
    VALID_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-000*1"]
    TEST_FILE= ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-000*1"]
elif DATA_MODE == 4:
    DATA_FILE = "/home/hadoop-hmart-waimaiad/cephfs/data/yangfan129/train_data/tfrecord-multi-channel-v1/"
    #TRAIN_LIST = ["20211222", "20211223", "20211224", "20211225"]
    TRAIN_LIST = ["20220123"]
    VALID_LIST = ["20220124"]
    TRAIN_FILE = [DATA_FILE + x + "/part-r-*" for x in TRAIN_LIST]
    VALID_FILE = [DATA_FILE + x + "/part-r-0001*" for x in VALID_LIST]
    TEST_FILE = [DATA_FILE + x + "/part-r-00011" for x in TRAIN_LIST]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MEAN_VAR_PATH_DELIVERY = "./avg_std/delivery"
MODEL_SAVE_PATH = "../model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
````

## File: rerank_model_dir/MultiG_E_model/model.py
````python
# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature

    def inter_attr_self_attention(self,input):

        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input,[-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output, self.inter_attr_self_attention_weights['weight_MLP'])

        return output

    def intra_attr_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V), (-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        return output

    def _oam(self,input):

        self.intra_attr_output = tf.identity(input)
        intra_attr_output_each_poi = tf.identity(input)
        for i in range(FEATURE_CATE_NUM):
            self.tmp_intra_attr_input = tf.reshape(tf.gather(input,list(range(i,i+1)),axis=2),[-1,POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_intra_attr_output = self.intra_attr_self_attention(self.tmp_intra_attr_input)
            self.tmp_intra_attr_output_mlp = tf.matmul(self.tmp_intra_attr_input + self.tmp_intra_attr_output, self.intra_attr_self_attention_weights['weight_MLP'])
            if i == 0:
                self.intra_attr_output = tf.expand_dims(self.tmp_intra_attr_output_mlp,axis=1)
                intra_attr_output_each_poi = tf.expand_dims(tf.reshape(self.tmp_intra_attr_output,[-1,POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]]),axis=2)
            else:
                self.intra_attr_output = tf.concat([self.intra_attr_output,tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)],axis=1)
                intra_attr_output_each_poi = tf.concat([intra_attr_output_each_poi,tf.expand_dims(tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)],axis=2)

        inter_attr_output = self.inter_attr_self_attention(self.intra_attr_output)

        # inter_attr_output: -1 * 8
        # intra_attr_output_each_poi : -1 * POI_NUM * feanum * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]
        return inter_attr_output,intra_attr_output_each_poi

    def _user_interest_target_attention(self,target,behavior,mask):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _user_interest_target_attention_topK(self,target,behavior,mask,k):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        behavior = tf.tile(behavior,[k,1,1])
        mask = tf.tile(mask,[k,1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _intra_poi_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
                           self.intra_poi_self_attention_weights['weight_MLP'])

        return output

    
    def positional_encoding(self,zero_pad=True,scale=True):

        N = 1
        T = 5
        num_units = 8

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # batch * 8

        position_enc = np.array([[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5



        return outputs


    def _PPML(self,features):
        position_encoding = features['position_encoding']
        position_encoding = tf.expand_dims(tf.expand_dims(position_encoding, axis=0),axis=0)
        self.ppml_page = tf.identity(self.behavior_cate_feature_embeddings)
        self.ppml_permuation = tf.identity(self.behavior_cate_feature_embeddings)
        # batch * 5 * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_behavior_input_embedding = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=3),[-1,PAGE_NUM, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_behavior_input_embedding = tf.reduce_mean(self.tmp_behavior_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_page = tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)
            else:
                self.ppml_page = tf.concat([self.ppml_page,tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)],axis=2)
        self.ppml_page = tf.reduce_mean(self.ppml_page,axis=2) # # batch * page * emb
        # batch * pe * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_permuation_input_embedding = tf.reshape(tf.gather(self.full_permuation_embeddings,list(range(i,i+1)),axis=3),[-1,PERMUATION_SIZE, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_permuation_input_embedding = tf.reduce_mean(self.tmp_permuation_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_permuation = tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)
            else:
                self.ppml_permuation = tf.concat([self.ppml_permuation,tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)],axis=2)
        self.ppml_permuation = tf.reduce_mean(self.ppml_permuation, axis=2)  # # batch * permuation * emb

    def _SimHash_TopOne(self,features):
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        top_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=1).indices # batch * k
        last_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=1).indices
        return top_permutation_indicces,last_permutation_indicces

    def _SimHash(self,features): #yhl:120 select topk(20)
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        # # select top40
        # top_30_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K + 10).indices # batch * k
        top_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K).indices # batch * k
        last_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=TOP_K).indices
        return top_k_permutation_indicces,last_k_permutation_indicces

    # 简化处理，使用全排列精排累积ctr的topK
    def BeamSearch(self):
        expose_rate = tf.expand_dims(tf.constant(EXPOSE_RATE_FOR_BEAM_SEARCH,tf.float32),axis=0)
        full_permuation_feat_predict = tf.reshape(self.full_permuation_feat_predict,[-1,PERMUATION_SIZE,POI_NUM]) * expose_rate
        cul_ctr = tf.reduce_sum(full_permuation_feat_predict,axis=2) # batch * permutation
        top_k_permutation_indicces = tf.nn.top_k(cul_ctr, k=TOP_K).indices  # batch * k
        return top_k_permutation_indicces

    def _FGSM(self,features):
        # get behavior and permuation emb
        self._PPML(features)
        self.top_k_permutation_indicces,last_k_permutation_indicces = self._SimHash(features)
        random_k_permutation_indicces = tf.tile(features['random_indices'],[tf.shape(self.top_k_permutation_indicces)[0],1])
        beam_k_permutation_indicces = self.BeamSearch()

        # top k
        self.top_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_k_permutation_indicces) # batch * k * 5 * 7 * 8
        # 在这个地方加scm,生成序列跟其他序列的距离
        self.top_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_embeddings = tf.stop_gradient(self.top_k_permuation_embeddings)
        self.top_k_permuation_predict_fea = tf.stop_gradient(self.top_k_permuation_predict_fea)
        self.top_k_permuation_mask = tf.stop_gradient(self.top_k_permuation_mask)

        # lask k
        self.last_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,last_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.last_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_embeddings = tf.stop_gradient(self.last_k_permuation_embeddings)
        self.last_k_permuation_predict_fea = tf.stop_gradient(self.last_k_permuation_predict_fea)
        self.last_k_permuation_mask = tf.stop_gradient(self.last_k_permuation_mask)

        # random k
        self.random_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,random_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.random_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,random_k_permutation_indicces)  # # batch * k * 5 * 1
        self.random_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,random_k_permutation_indicces)  # # batch * k * 5 * 1

        # beam search k
        self.beam_search_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,beam_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.beam_search_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,beam_k_permutation_indicces)  # # batch * k * 5 * 1
        self.beam_search_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,beam_k_permutation_indicces)  # # batch * k * 5 * 1
    
    #单点分布Pointwise(indp)
    def _Pointwise_generator_model(self, features, labels, mode, params):
        # get behavior and permuation emb
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)
        self._PPML(features)
        self.top_permutation_indicces,last_permutation_indicces = self._SimHash_TopOne(features)

        # top k
        self.top_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_permutation_indicces) # [bs, 1, 5, 7, 8]
        self.top_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_permutation_indicces) # # batch * 1 * 5 * 1
        self.top_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_permutation_indicces) # # batch * k * 5 * 1
        target_page_predict_fea = tf.reshape(self.top_permuation_predict_fea,[-1, POI_NUM, 1])

        self.top_permuation_embeddings = tf.stop_gradient(self.top_permuation_embeddings)
        self.top_permuation_predict_fea = tf.stop_gradient(self.top_permuation_predict_fea)
        self.top_permuation_mask = tf.stop_gradient(self.top_permuation_mask)

        pointwise_ctr_out = tf.layers.dense(target_page_predict_fea, OUT_NUM, activation=None, name="pointwise_final_out_ctr",reuse=tf.AUTO_REUSE)


        if not self.train:
                pointwise_ctr_out = tf.nn.sigmoid(pointwise_ctr_out)

        self.pointwise_generator_out = tf.reshape(pointwise_ctr_out,[-1,POI_NUM]) # bs*5
        # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
        # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
        # 使用 tf.argsort 来获取排序后的索引
        self.pointwise_generator_permution_index = tf.argsort(self.pointwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
        self.pointwise_generator_permution_index = tf.stop_gradient(self.pointwise_generator_permution_index)

        self.pointwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.pointwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.pointwise_generator_embedding = tf.expand_dims(self.pointwise_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
        self.pointwise_generator_embedding = tf.stop_gradient(self.pointwise_generator_embedding)

        self.pointwise_generator_permution_index_expand = tf.expand_dims(self.pointwise_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
        self.pointwise_generator_permution_index_expand = tf.stop_gradient(self.pointwise_generator_permution_index_expand)

        self.pointwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.pointwise_generator_permution_index),[-1, POI_NUM, 1])
        self.pointwise_permuation_mask = tf.expand_dims(self.pointwise_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
        self.pointwise_permuation_mask = tf.stop_gradient(self.pointwise_permuation_mask)
        # self.Q_network_output = self.out 

    
    #位置分布ListWise
    def _Listwise_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            listwise_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.listwise_cur_page_oam_output,self.listwise_intra_attr_output_each_poi = self._oam(listwise_cate_feature_embeddings_stopped)


            self.listwise_behavior_page_oam_output = tf.identity(self.listwise_cur_page_oam_output)
            for i in range(PAGE_NUM):
                listwise_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.listwise_tmp_behavior_input = tf.reshape(tf.gather(listwise_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_behavior_oam_output,_ = self._oam(self.listwise_tmp_behavior_input)
                if i == 0:
                    self.listwise_behavior_page_oam_output = tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)
                else:
                    self.listwise_behavior_page_oam_output = tf.concat([self.listwise_behavior_page_oam_output,tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            listwise_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.listwise_user_interest_emb = self._user_interest_target_attention(self.listwise_cur_page_oam_output,self.listwise_behavior_page_oam_output,listwise_mask_feature_stopped)
            
            #features['page_mask'] noindp

            self.listwise_poi_rep_emb = self.listwise_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.listwise_tmp_poi_input = tf.reshape(tf.gather(self.listwise_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_poi_output = self._intra_poi_self_attention(self.listwise_tmp_poi_input)
                if i == 0:
                    self.listwise_poi_rep_emb = tf.expand_dims(self.listwise_tmp_poi_output, axis=1)
                else:
                    self.listwise_poi_rep_emb = tf.concat([self.listwise_poi_rep_emb, tf.expand_dims(self.listwise_tmp_poi_output, axis=1)], axis=1)

            self.listwise_cur_page_oam_output = tf.tile(tf.expand_dims(self.listwise_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.listwise_user_interest_emb = tf.tile(tf.expand_dims(self.listwise_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            listwise_fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Listwise_MLP_A" + str(i)
                listwise_fc_out = tf.layers.dense(listwise_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                listwise_fc_out = tf.nn.swish(listwise_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            listwise_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            listwise_fc_input = tf.concat([self.listwise_cur_page_oam_output, self.listwise_user_interest_emb, self.listwise_poi_rep_emb, listwise_feat_predict_stopped,listwise_fc_out], axis=2)

            listwise_fc_out_ctr, listwise_fc_out_imp = tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "Listwise_MLP_B" + str(i)
                listwise_fc_out_ctr = tf.layers.dense(listwise_fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                listwise_fc_out_ctr = tf.nn.relu(listwise_fc_out_ctr)

            listwise_ctr_out = tf.layers.dense(listwise_fc_out_ctr, OUT_NUM, activation=None, name="listwise_final_out_ctr")

            if not self.train:
                listwise_ctr_out = tf.nn.sigmoid(listwise_ctr_out)

            self.listwise_generator_out = tf.reshape(listwise_ctr_out,[-1,POI_NUM]) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.listwise_generator_permution_index = tf.argsort(self.listwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.listwise_generator_permution_index = tf.stop_gradient(self.listwise_generator_permution_index)

            self.listwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.listwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.listwise_generator_embedding = tf.expand_dims(self.listwise_generator_embedding, axis=1)  # [bs, 1, 5,7,8]
            self.listwise_generator_embedding = tf.stop_gradient(self.listwise_generator_embedding)

            self.listwise_generator_permution_index_expand = tf.expand_dims(self.listwise_generator_permution_index, axis=1)  # [bs, 1, 5]
            self.listwise_generator_permution_index_expand = tf.stop_gradient(self.listwise_generator_permution_index_expand)

            self.listwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.listwise_generator_permution_index),[-1, POI_NUM, 1])
            self.listwise_permuation_mask = tf.expand_dims(self.listwise_permuation_mask, axis=1)  # [bs,1,5,1]
            self.listwise_permuation_mask = tf.stop_gradient(self.listwise_permuation_mask)
            # self.Q_network_output = self.out 
    
    #生成式ListWise
    def _headlayers(self, top_tensor_list, position_tensor):
        """
        top_tensor_list: [batch_size, 50 embedding_size]]
        position_tensor: [batch_size , embedding_size]
        output: [batch_size, predict_len, 1]
        """
        position_tensor = tf.tile(tf.expand_dims(position_tensor, axis=1), [1, POI_NUM, 1])
        output = tf.concat([top_tensor_list, position_tensor], axis=2)
        with tf.variable_scope("head_layers", reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(output, 64, activation=tf.nn.relu, name="head_1")
            output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="head_2")
            output = tf.layers.dense(output, 1, activation=None, name="score")
        return output

    def _Generative_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.generative_cur_page_oam_output,self.generative_intra_attr_output_each_poi = self._oam(generative_cate_feature_embeddings_stopped)

            self.generative_behavior_page_oam_output = tf.identity(self.generative_cur_page_oam_output)
            for i in range(PAGE_NUM):
                generative_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.generative_tmp_behavior_input = tf.reshape(tf.gather(generative_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_behavior_oam_output,_ = self._oam(self.generative_tmp_behavior_input)
                if i == 0:
                    self.generative_behavior_page_oam_output = tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)
                else:
                    self.generative_behavior_page_oam_output = tf.concat([self.generative_behavior_page_oam_output,tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            generative_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.generative_user_interest_emb = self._user_interest_target_attention(self.generative_cur_page_oam_output,self.generative_behavior_page_oam_output,generative_mask_feature_stopped)

            self.generative_poi_rep_emb = self.generative_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.generative_tmp_poi_input = tf.reshape(tf.gather(self.generative_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_poi_output = self._intra_poi_self_attention(self.generative_tmp_poi_input)
                if i == 0:
                    self.generative_poi_rep_emb = tf.expand_dims(self.generative_tmp_poi_output, axis=1)
                else:
                    self.generative_poi_rep_emb = tf.concat([self.generative_poi_rep_emb, tf.expand_dims(self.generative_tmp_poi_output, axis=1)], axis=1)

            self.generative_cur_page_oam_output = tf.tile(tf.expand_dims(self.generative_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.generative_user_interest_emb = tf.tile(tf.expand_dims(self.generative_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            generative_fc_out = tf.reshape(generative_cate_feature_embeddings_stopped,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Generative_MLP_A" + str(i)
                generative_fc_out = tf.layers.dense(generative_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                generative_fc_out = tf.nn.swish(generative_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            generative_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            generative_fc_input = tf.concat([self.generative_cur_page_oam_output, self.generative_user_interest_emb, self.generative_poi_rep_emb, generative_feat_predict_stopped,generative_fc_out], axis=2) # Batch_size * POI_NUM * FEAT_NUM
            generative_fc_input = tf.layers.dense(generative_fc_input, 64, activation=None, name="generative_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            batch_size      = tf.shape(generative_fc_input)[0]
            with tf.variable_scope("rnn_init_state", reuse=tf.AUTO_REUSE):
                state = tf.reduce_mean(generative_fc_input, axis=1)  # [B,64]
                state = tf.stop_gradient(state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                input_tensor = tf.zeros_like(state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64, name='gru_cell')  # 替换 tf.contrib.rnn.GRUCell
        
            if self.train:
                self.label = labels['ctr_label']
                self.generative_generator_loss = 0.0
                output_list = []
                label_list = []
                eval_weight_list = []
                generative_generator_out = []
                generative_generator_permution_index = []
                mask_tmp = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]
                eval_ones = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 5,1], head_layer最后一层不用激活函数
                    head_score_origin = tf.reshape(head_score, [-1, POI_NUM]) #[B, 5]
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - mask_tmp, -1.0e9), head_score_origin)) #[B,5]
                    # lable
                    select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1), [tf.shape(head_score)[0], 1]) #[B, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32) #[B, 5]

                    # sample_reward = realshow_list * sample_weight
                    label_mask = self.label * select_onehot #[B, 5], 只有曝光位置有reward, 其他为 0
                    # loss
                    tmp_loss = -1.0 * tf.reduce_sum(label_mask * tf.log(tf.clip_by_value(head_score, 1e-9, 1.0)), axis=-1) # [B]
                    tmp_loss = tf.reduce_mean(tmp_loss)
                    self.generative_generator_loss = self.generative_generator_loss + tmp_loss
                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
                    

                    output_list.append(head_score)
                    label_list.append(label_mask)
                    weight = mask_tmp * eval_ones
                    eval_weight_list.append(weight)

                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [B, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]
                    input_tensor = tf.stop_gradient(input_tensor)
            else:
                generative_generator_permution_index = []
                generative_generator_out = []
                mask_tmp = tf.ones([batch_size, PAGE_NUM], dtype=tf.float32) # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 50,1]
                    head_score = tf.reshape(head_score, [-1, POI_NUM])
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), head_score)) #[B,50]
   
                    select_index = tf.reshape(tf.argmax(head_score, axis=1, output_type=tf.int32), [-1, 1])  # [batch_size, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32)
                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [batch_size, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]

                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
            


                

            self.generative_generator_out = tf.stack(generative_generator_out, axis=1) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.generative_generator_permution_index = tf.stack(generative_generator_permution_index, axis=1) # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.generative_generator_permution_index = tf.stop_gradient(self.generative_generator_permution_index)

            self.generative_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.generative_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.generative_generator_embedding = tf.expand_dims(self.generative_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.generative_generator_embedding = tf.stop_gradient(self.generative_generator_embedding)

            self.generative_generator_permution_index_expand = tf.expand_dims(self.generative_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
            self.generative_generator_permution_index_expand = tf.stop_gradient(self.generative_generator_permution_index_expand)

            self.generative_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.generative_generator_permution_index),[-1, POI_NUM, 1])
            self.generative_permuation_mask = tf.expand_dims(self.generative_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
            self.generative_permuation_mask = tf.stop_gradient(self.generative_permuation_mask)
            # self.Q_network_output = self.out
    
    def _OCPM(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.cur_page_oam_output,self.intra_attr_output_each_poi = self._oam(self.cate_feature_embeddings)

            self.behavior_page_oam_output = tf.identity(self.cur_page_oam_output)
            for i in range(PAGE_NUM):
                self.tmp_behavior_input = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_behavior_oam_output,_ = self._oam(self.tmp_behavior_input)
                if i == 0:
                    self.behavior_page_oam_output = tf.expand_dims(self.tmp_behavior_oam_output,axis=1)
                else:
                    self.behavior_page_oam_output = tf.concat([self.behavior_page_oam_output,tf.expand_dims(self.tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            self.user_interest_emb = self._user_interest_target_attention(self.cur_page_oam_output,self.behavior_page_oam_output,features['page_mask'])

            self.poi_rep_emb = self.intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.tmp_poi_input = tf.reshape(tf.gather(self.intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_poi_output = self._intra_poi_self_attention(self.tmp_poi_input)
                if i == 0:
                    self.poi_rep_emb = tf.expand_dims(self.tmp_poi_output, axis=1)
                else:
                    self.poi_rep_emb = tf.concat([self.poi_rep_emb, tf.expand_dims(self.tmp_poi_output, axis=1)], axis=1)

            self.cur_page_oam_output = tf.tile(tf.expand_dims(self.cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.user_interest_emb = tf.tile(tf.expand_dims(self.user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat([self.cur_page_oam_output, self.user_interest_emb, self.poi_rep_emb, self.feat_predict,fc_out], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_model_final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out,[-1,POI_NUM])
            self.Q_network_output = self.out

    def _OCPM_TOP_K(self, features, labels, mode, params,k_page,k_predict,k,mask):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        target_page_embedding = tf.reshape(k_page,[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        target_page_predict_fea = tf.reshape(k_predict,[-1, POI_NUM, 1])

        with tf.name_scope('dnn_model'):
            cur_page_oam_output, target_intra_attr_output_each_poi = self._oam(target_page_embedding)

            behavior_page_oam_output = self.behavior_page_oam_output

            user_interest_emb = self._user_interest_target_attention_topK(cur_page_oam_output,behavior_page_oam_output,features['page_mask'],k)

            poi_rep_emb = target_intra_attr_output_each_poi # （batch * k） * poi * 8
            for i in range(POI_NUM):
                tmp_poi_input = tf.reshape(
                    tf.gather(target_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                tmp_poi_output = self._intra_poi_self_attention(tmp_poi_input)
                if i == 0:
                    poi_rep_emb = tf.expand_dims(tmp_poi_output, axis=1)
                else:
                    poi_rep_emb = tf.concat(
                        [poi_rep_emb, tf.expand_dims(tmp_poi_output, axis=1)], axis=1)

            cur_page_oam_output = tf.tile(tf.expand_dims(cur_page_oam_output, axis=1),[1, POI_NUM, 1])
            user_interest_emb = tf.tile(tf.expand_dims(user_interest_emb, axis=1), [1, POI_NUM, 1])

            fc_out = tf.reshape(target_page_embedding, [-1, POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out = tf.nn.swish(fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat(
                [cur_page_oam_output, user_interest_emb, poi_rep_emb, target_page_predict_fea, fc_out],axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input,[-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i],
                                             activation=None, name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_topk_final_out_ctr",reuse=tf.AUTO_REUSE)
            ctr_out = tf.nn.sigmoid(ctr_out)

            topk_out = tf.reshape(ctr_out, [-1, k, POI_NUM])
            mask = tf.reshape(mask,[-1,k,POI_NUM])
            return topk_out * mask

    


    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss
            self.pointwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.pointwise_generator_out, weights=self.mask))
            self.listwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.listwise_generator_out, weights=self.mask))
            # self.submodularity_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.submodularity_generator_out, weights=self.mask))

            self.evaluator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.out, weights=self.mask))
            self.loss = self.pointwise_generator_loss + self.listwise_generator_loss  + self.evaluator_loss + self.generative_generator_loss
            # if USE_CONSTRATIVE_LOSS:
            #     self.loss -= CONSTRATIVE_LOSS_K * (tf.reduce_mean(self.topK_output,axis=[0,1,2]) - tf.reduce_mean(self.lastK_output,axis=[0,1,2]))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    
    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer(),
                                            )
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.intra_attr_self_attention_weights = {
                'weight_Q':tf.get_variable(name='intra_attr_self_attention_weights_Q',shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1],CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                           initializer=tf.random_normal_initializer(0.0, glorot),
                                           dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_attr_self_attention_weights_MLP',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32)
            }

            self.inter_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='inter_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.intra_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_poi_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_poi_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_poi_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_poi_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        cate_fea = tf.reshape(features['cate_feature'],[-1,POI_NUM * FEATURE_CATE_NUM])
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], cate_fea),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        full_permuation_index = tf.reshape(features['full_permuation_index'],[PERMUATION_SIZE,POI_NUM])

        self.cate_feature_embeddings_for_permuation = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.full_permuation_index = tf.reshape(tf.tile(tf.expand_dims(full_permuation_index,axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0],1,1]),[-1,POI_NUM * PERMUATION_SIZE])
        self.full_permuation_embeddings = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        
        # self.input_embedding = tf.concat(
        #     [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']
        self.feat_predict_for_permuation = tf.reshape(self.feat_predict,[-1,POI_NUM,1])

        self.full_permuation_feat_predict = tf.reshape(tf.batch_gather(self.feat_predict_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,1])


        self.permuation_mask = tf.reshape(features['permuation_mask'],[-1,POI_NUM,1])
        self.full_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.full_permuation_index),[-1, PERMUATION_SIZE, POI_NUM, 1])


        behavior_cate_fea = tf.reshape(features['behavior_cate_feature'],[-1,PAGE_NUM * POI_NUM * FEATURE_CATE_NUM])
        self.behavior_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], behavior_cate_fea),
            [-1, PAGE_NUM , POI_NUM, FEATURE_CATE_NUM ,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        self.behavior_input_embedding = tf.concat([self.behavior_cate_feature_embeddings], axis=3)
        self.behavior_feat_predict = features['behavior_dense_feature']

        return features

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''
        def format_log(tensors):
            log0 = "train info: step {}, loss={:.4f}, pointwise_loss={:.4f},listwise_generator_loss={:.4f},generative_generator_loss={:.4f},evaluator_loss={:.4f}, " \
                   "pointwise_ctr_out={:.4f}, listwise_ctr_out={:.4f}, generative_ctr_out={:.4f}, ctr_out={:.4f},  avg_ctr_topk={:.4f},  avt_ctr_lastk={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["pointwise_loss"], tensors["listwise_generator_loss"], tensors["generative_generator_loss"], tensors["evaluator_loss"],
                tensors["pointwise_ctr_out"], tensors["listwise_ctr_out"], tensors["generative_ctr_out"], tensors["ctr_out"],tensors['avg_ctr_topk'],tensors['avt_ctr_lastk']
            )
            return log0
        
        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "pointwise_loss": self.pointwise_generator_loss,
                                                        "listwise_generator_loss": self.listwise_generator_loss,
                                                        "generative_generator_loss":self.generative_generator_loss,
                                                        "evaluator_loss":self.evaluator_loss,
                                                        "pointwise_ctr_out":tf.reduce_mean(self.pointwise_generator_out),
                                                        "listwise_ctr_out":tf.reduce_mean(self.listwise_generator_out),
                                                        "generative_ctr_out":tf.reduce_mean(self.generative_generator_out),
                                                        "ctr_out": ctr_out,
                                                        "avg_ctr_topk" : tf.reduce_mean(self.topK_output,axis=[0,1,2]),
                                                        "avt_ctr_lastk" : tf.reduce_mean(self.lastK_output,axis=[0,1,2]),
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        self._FGSM(features)
        #pointwise输出一个序列
        self._Pointwise_generator_model(features, labels, mode, params)
        #位置分布输出一个序列
        self._Listwise_generator_model(features, labels, mode, params)
        #GRN输出一个序列
        self._Generative_generator_model(features, labels, mode, params)
        
        generator_num = 3
        self._OCPM(features, labels, mode, params)
        self.generator_permuation_embeddings = tf.concat([self.pointwise_generator_embedding,self.listwise_generator_embedding,self.generative_generator_embedding],axis =1) # [bs, k, 5,7,8]
        self.generator_permuation_embeddings = tf.stop_gradient(self.generator_permuation_embeddings)

        self.pointwise_generator_out_expand = tf.expand_dims(self.pointwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.listwise_generator_out_expand = tf.expand_dims(self.listwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generative_generator_out_expand = tf.expand_dims(self.generative_generator_out, axis=1)  # 形状变为 [bs, 1, 5]

        self.generator_permuation_predict_fea = tf.concat([self.pointwise_generator_out_expand,self.listwise_generator_out_expand,self.generative_generator_out_expand],axis =1) # [bs, k, 5]
        self.generator_permuation_predict_fea = tf.stop_gradient(self.generator_permuation_predict_fea)

        self.generator_permuation_index = tf.concat([self.pointwise_generator_permution_index_expand,self.listwise_generator_permution_index_expand,self.generative_generator_permution_index_expand],axis =1)
        self.generator_permuation_index = tf.stop_gradient(self.generator_permuation_index)

        self.generator_permuation_mask = tf.concat([self.pointwise_permuation_mask,self.listwise_permuation_mask,self.generative_permuation_mask],axis =1) # [bs, k, 5,1]
        self.generator_permuation_mask = tf.stop_gradient(self.generator_permuation_mask)


        

        self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generator_permuation_predict_fea,generator_num,self.generator_permuation_mask)
        self.lastK_output = self._OCPM_TOP_K(features, labels, mode, params, self.last_k_permuation_embeddings, self.last_k_permuation_predict_fea, TOP_K,self.last_k_permuation_mask)
        self.randK_output = self._OCPM_TOP_K(features, labels, mode, params, self.random_k_permuation_embeddings, self.random_k_permuation_predict_fea, TOP_K,self.random_k_permuation_mask)
        self.beamK_output = self._OCPM_TOP_K(features, labels, mode, params, self.beam_search_k_permuation_embeddings, self.beam_search_k_permuation_predict_fea, TOP_K,self.beam_search_k_permuation_mask)
        self.allK_output = self._OCPM_TOP_K(features, labels, mode, params, self.full_permuation_embeddings, self.full_permuation_feat_predict, PERMUATION_SIZE,self.full_permuation_mask)


        


        # 输出avg-ctr最高的序列
        self.avg_ctr_of_topK = tf.reduce_mean(self.topK_output,axis=2)
        max_ctr_index_in_topk = tf.nn.top_k(self.avg_ctr_of_topK,k=1).indices # batch * 1
        self.final_rerank_output_index = tf.batch_gather(self.generator_permuation_index,max_ctr_index_in_topk) # bs,1,5
        # self.max_ctr_index = tf.batch_gather(self.top_k_permutation_indicces,self.max_ctr_index_in_topk) # batch * 1
        # full_permuation_index = tf.reshape(features['full_permuation_index'], [PERMUATION_SIZE, POI_NUM])
        # full_permuation_index = tf.tile(tf.expand_dims(full_permuation_index, axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0], 1, 1]) # batch * 120 * 5

        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out"),
                    'output_index':tf.identity(self.final_rerank_output_index,"out_index")
                    }
            else:
                ctr_out = self.out
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                           "ctr_topk": self.topK_output,
                           "ctr_lastk": self.lastK_output,
                           "ctr_randk": self.randK_output,
                           "ctr_all": self.allK_output,
                           "ctr_beamk": self.beamK_output,
                           'out_index':self.final_rerank_output_index
                         }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)


    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model/config.py
````python
# coding: utf-8 -*-
import shutil
import time
import os

# result
RANDOM_SEED = 2022
BATCH_SIZE = 1024
IMP_LOSS_WEIGHT = 0.02
# basic config
EPOCH = 1
LEARNING_RATE = 0.001
DATA_MODE = 3 # 1:local train，2:local test, 3:docker evaluate
TRAIN_MODE = 3
MODEL_NAME = "MultiGE_submodularity"

USE_CONSTRATIVE_LOSS = True
CONSTRATIVE_LOSS_K = 0.01

# poi类别特征
FEATURE_CATE_NUM = 7 # v1r3:19
# dense特征
FEATURE_DENSE_NUM = 5  # v1:28 v1r2:79 v1r3:83
# 预估值特征
FEATURE_CXR_NUM = 3
# 环境特征
FEATURE_ENV_NUM = 2
# 自然poi
FEATURE_NATURE_POI = 25

# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# hash vector
HASH_VECTOR = [[0.4011729380078948, 0.9907933488085712, 0.889170658914898, 0.015553701343792192, 0.2269008585804, 0.889609750280199, 0.19280345796299014, 0.33362195188257815],
               [0.1174580997476552, 0.4475570392889796, 0.9925939893135071, 0.7296640075216434, 0.6436313332140967, 0.8568068597644793, 0.9018604021483339, 0.28774343258170776],
               [0.4374993384759095, 0.6807149381274915, 0.20502126763268802, 0.6968687323602859, 0.6449029002244834, 0.08732080642447282, 0.6119476780855001, 0.8616788453789646],
               [0.19344266090314144, 0.4268233179146762, 0.10951597767480326, 0.9867838283258178, 0.8340011944969644, 0.7992329879482085, 0.005303560724105649, 0.9662924610057512],
               [0.5824282763301396, 0.5090708710943849, 0.4462703076663568, 0.5482943153972023, 0.06782829736851825, 0.08907408658464577, 0.13400486343251583, 0.1848228429919272]]

POSITION_ENCODING = [[0.08790239841717873, 0.033267486152506076, 0.5495130189114207, 0.17419777583517537, 0.6838981992197484, 0.07935154925635501, 0.02665372302227631, 0.4181008411574786],
                     [0.02038159041970189, 0.9310485215006198, 0.723661313625571, 0.5110323516977285, 0.7812851438212606, 0.40722710713947474, 0.04646490014890503, 0.9565353323381218],
                     [0.17279361698259843, 0.12968275664201512, 0.3183103529758954, 0.6360591081256931, 0.1558507653689548, 0.5972802646455662, 0.4380619835390329, 0.6088094249662641],
                     [0.038262298606345335, 0.9999545626217287, 0.7113776275017341, 0.5434618368150265, 0.6853527957705402, 0.9662653254145415, 0.9641592716989676, 0.3443813983264],
                     [0.7903243938847678, 0.9952713339078417, 0.8741415264071601, 0.45665348276461737, 0.7693872696125916, 0.772509599868299, 0.2540369924156157, 0.24781240400239857]]


TIME_AWARE_WEIGHT = [[1/2,1/2,1/3,1/3,1/4]]
PERMUATION_SIZE = 120
TOP_K = 20


EXPOSE_RATE_FOR_BEAM_SEARCH = [[1.0,0.9,0.8,0.7,0.6]]

# N: Cut Number of POI For Train
POI_NUM = 5
FEATURE_NUM = 9
PAGE_NUM = 5
FEATURE_NUM_FOR_PAGE = 11

# 属性特征：KA AOR BRAND
FEATURE_ATTR_NUM = 3

# DELIVERY_FEAT
DELIVERY_FEAT_NUM = 4

# OUT NUM
OUT_NUM = 1

PLACE_HOLDER_NUM = 11
DENSE_FEAT_NUM = 439


# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 20],
    'INPUT_TENSOR_LAYERS_B': [128, 32],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
# A_INPUT_DIM = POI_NUM * (MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)
MLP_INPUT_DIM = CATE_FEATURE_EMBEDDINGS_SHAPE[1] * 3 + 1 + MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1]
DIN_CONF = {}

# train data
# /users/lemonace/Downloads/tfrecord-rl-limit5-v1
if DATA_MODE == 1:
    TRAIN_FILE = ['/users/meituan_sxw/Downloads/part-r-00046']
    VALID_FILE = TRAIN_FILE
    PREDICT_FILE = VALID_FILE
    TEST_FILE = PREDICT_FILE
elif DATA_MODE == 2:
    TRAIN_FILE = ['/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/test_data/part-r-*']
    VALID_FILE = TRAIN_FILE
    TEST_FILE = VALID_FILE
elif DATA_MODE == 3:
    TRAIN_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/train_data/part-r-*"]
    VALID_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-*"]
    TEST_FILE= ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-*"]
elif DATA_MODE == 4:
    DATA_FILE = "/home/hadoop-hmart-waimaiad/cephfs/data/yangfan129/train_data/tfrecord-multi-channel-v1/"
    #TRAIN_LIST = ["20211222", "20211223", "20211224", "20211225"]
    TRAIN_LIST = ["20220123"]
    VALID_LIST = ["20220124"]
    TRAIN_FILE = [DATA_FILE + x + "/part-r-*" for x in TRAIN_LIST]
    VALID_FILE = [DATA_FILE + x + "/part-r-0001*" for x in VALID_LIST]
    TEST_FILE = [DATA_FILE + x + "/part-r-00011" for x in TRAIN_LIST]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MEAN_VAR_PATH_DELIVERY = "./avg_std/delivery"
MODEL_SAVE_PATH = "./model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model/model.py
````python
# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature

    def inter_attr_self_attention(self,input):

        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input,[-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output, self.inter_attr_self_attention_weights['weight_MLP'])

        return output

    def intra_attr_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V), (-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        return output

    def _oam(self,input):

        self.intra_attr_output = tf.identity(input)
        intra_attr_output_each_poi = tf.identity(input)
        for i in range(FEATURE_CATE_NUM):
            self.tmp_intra_attr_input = tf.reshape(tf.gather(input,list(range(i,i+1)),axis=2),[-1,POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_intra_attr_output = self.intra_attr_self_attention(self.tmp_intra_attr_input)
            self.tmp_intra_attr_output_mlp = tf.matmul(self.tmp_intra_attr_input + self.tmp_intra_attr_output, self.intra_attr_self_attention_weights['weight_MLP'])
            if i == 0:
                self.intra_attr_output = tf.expand_dims(self.tmp_intra_attr_output_mlp,axis=1)
                intra_attr_output_each_poi = tf.expand_dims(tf.reshape(self.tmp_intra_attr_output,[-1,POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]]),axis=2)
            else:
                self.intra_attr_output = tf.concat([self.intra_attr_output,tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)],axis=1)
                intra_attr_output_each_poi = tf.concat([intra_attr_output_each_poi,tf.expand_dims(tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)],axis=2)

        inter_attr_output = self.inter_attr_self_attention(self.intra_attr_output)

        # inter_attr_output: -1 * 8
        # intra_attr_output_each_poi : -1 * POI_NUM * feanum * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]
        return inter_attr_output,intra_attr_output_each_poi

    def _user_interest_target_attention(self,target,behavior,mask):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _user_interest_target_attention_topK(self,target,behavior,mask,k):
        target = tf.tile(tf.expand_dims(target,axis=1),[1,tf.shape(behavior)[1],1])
        behavior = tf.tile(behavior,[k,1,1])
        mask = tf.tile(mask,[k,1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights,axis=1) # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights,axis=2) * behavior,axis=1)
        return user_interest # batch * 8

    def _intra_poi_self_attention(self,input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,(tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
                           self.intra_poi_self_attention_weights['weight_MLP'])

        return output

    
    def positional_encoding(self,zero_pad=True,scale=True):

        N = 1
        T = 5
        num_units = 8

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # batch * 8

        position_enc = np.array([[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5



        return outputs


    def _PPML(self,features):
        position_encoding = features['position_encoding']
        position_encoding = tf.expand_dims(tf.expand_dims(position_encoding, axis=0),axis=0)
        self.ppml_page = tf.identity(self.behavior_cate_feature_embeddings)
        self.ppml_permuation = tf.identity(self.behavior_cate_feature_embeddings)
        # batch * 5 * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_behavior_input_embedding = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=3),[-1,PAGE_NUM, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_behavior_input_embedding = tf.reduce_mean(self.tmp_behavior_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_page = tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)
            else:
                self.ppml_page = tf.concat([self.ppml_page,tf.expand_dims(self.tmp_behavior_input_embedding,axis=2)],axis=2)
        self.ppml_page = tf.reduce_mean(self.ppml_page,axis=2) # # batch * page * emb
        # batch * pe * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_permuation_input_embedding = tf.reshape(tf.gather(self.full_permuation_embeddings,list(range(i,i+1)),axis=3),[-1,PERMUATION_SIZE, POI_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_permuation_input_embedding = tf.reduce_mean(self.tmp_permuation_input_embedding * position_encoding,axis=2) # batch * page * emb
            if i == 0:
                self.ppml_permuation = tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)
            else:
                self.ppml_permuation = tf.concat([self.ppml_permuation,tf.expand_dims(self.tmp_permuation_input_embedding,axis=2)],axis=2)
        self.ppml_permuation = tf.reduce_mean(self.ppml_permuation, axis=2)  # # batch * permuation * emb

    def _SimHash_TopOne(self,features):
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        top_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=1).indices # batch * k
        last_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=1).indices
        return top_permutation_indicces,last_permutation_indicces

    def _SimHash(self,features): #yhl:120 select topk(20)
        page_hash_vector = tf.expand_dims(features['hash_vector'],axis=0) # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector) # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature,axis=1),[1,PERMUATION_SIZE,1,1]) # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'],axis=0),axis=0) # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation,axis=2),[1,1,PAGE_NUM,1])
        self.permuation_hash_signature = tf.sign(self.ppml_permuation * permuation_hash_vector) # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),axis=3) # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(tf.expand_dims(features['disantce_weight'],axis=0) * hamming_distance,axis=2) # batch * permution
        # # select top40
        # top_30_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K + 10).indices # batch * k
        top_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K).indices # batch * k
        last_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance,k=TOP_K).indices
        return top_k_permutation_indicces,last_k_permutation_indicces

    # 简化处理，使用全排列精排累积ctr的topK
    def BeamSearch(self):
        expose_rate = tf.expand_dims(tf.constant(EXPOSE_RATE_FOR_BEAM_SEARCH,tf.float32),axis=0)
        full_permuation_feat_predict = tf.reshape(self.full_permuation_feat_predict,[-1,PERMUATION_SIZE,POI_NUM]) * expose_rate
        cul_ctr = tf.reduce_sum(full_permuation_feat_predict,axis=2) # batch * permutation
        top_k_permutation_indicces = tf.nn.top_k(cul_ctr, k=TOP_K).indices  # batch * k
        return top_k_permutation_indicces

    def _FGSM(self,features):
        # get behavior and permuation emb
        self._PPML(features)
        self.top_k_permutation_indicces,last_k_permutation_indicces = self._SimHash(features)
        random_k_permutation_indicces = tf.tile(features['random_indices'],[tf.shape(self.top_k_permutation_indicces)[0],1])
        beam_k_permutation_indicces = self.BeamSearch()

        # top k
        self.top_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_k_permutation_indicces) # batch * k * 5 * 7 * 8
        # 在这个地方加scm,生成序列跟其他序列的距离
        self.top_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_k_permutation_indicces) # # batch * k * 5 * 1
        self.top_k_permuation_embeddings = tf.stop_gradient(self.top_k_permuation_embeddings)
        self.top_k_permuation_predict_fea = tf.stop_gradient(self.top_k_permuation_predict_fea)
        self.top_k_permuation_mask = tf.stop_gradient(self.top_k_permuation_mask)

        # lask k
        self.last_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,last_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.last_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_embeddings = tf.stop_gradient(self.last_k_permuation_embeddings)
        self.last_k_permuation_predict_fea = tf.stop_gradient(self.last_k_permuation_predict_fea)
        self.last_k_permuation_mask = tf.stop_gradient(self.last_k_permuation_mask)

        # random k
        self.random_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,random_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.random_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,random_k_permutation_indicces)  # # batch * k * 5 * 1
        self.random_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,random_k_permutation_indicces)  # # batch * k * 5 * 1

        # beam search k
        self.beam_search_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,beam_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.beam_search_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,beam_k_permutation_indicces)  # # batch * k * 5 * 1
        self.beam_search_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,beam_k_permutation_indicces)  # # batch * k * 5 * 1
    
    #单点分布Pointwise(indp)
    def _Pointwise_generator_model(self, features, labels, mode, params):
        # get behavior and permuation emb
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)
        self._PPML(features)
        self.top_permutation_indicces,last_permutation_indicces = self._SimHash_TopOne(features)

        # top k
        self.top_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,self.top_permutation_indicces) # [bs, 1, 5, 7, 8]
        self.top_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,self.top_permutation_indicces) # # batch * 1 * 5 * 1
        self.top_permuation_mask = tf.batch_gather(self.full_permuation_mask,self.top_permutation_indicces) # # batch * k * 5 * 1
        target_page_predict_fea = tf.reshape(self.top_permuation_predict_fea,[-1, POI_NUM, 1])

        self.top_permuation_embeddings = tf.stop_gradient(self.top_permuation_embeddings)
        self.top_permuation_predict_fea = tf.stop_gradient(self.top_permuation_predict_fea)
        self.top_permuation_mask = tf.stop_gradient(self.top_permuation_mask)

        pointwise_ctr_out = tf.layers.dense(target_page_predict_fea, OUT_NUM, activation=None, name="pointwise_final_out_ctr",reuse=tf.AUTO_REUSE)


        if not self.train:
                pointwise_ctr_out = tf.nn.sigmoid(pointwise_ctr_out)

        self.pointwise_generator_out = tf.reshape(pointwise_ctr_out,[-1,POI_NUM]) # bs*5
        # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
        # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
        # 使用 tf.argsort 来获取排序后的索引
        self.pointwise_generator_permution_index = tf.argsort(self.pointwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
        self.pointwise_generator_permution_index = tf.stop_gradient(self.pointwise_generator_permution_index)

        self.pointwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.pointwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.pointwise_generator_embedding = tf.expand_dims(self.pointwise_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
        self.pointwise_generator_embedding = tf.stop_gradient(self.pointwise_generator_embedding)

        self.pointwise_generator_permution_index_expand = tf.expand_dims(self.pointwise_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
        self.pointwise_generator_permution_index_expand = tf.stop_gradient(self.pointwise_generator_permution_index_expand)

        self.pointwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.pointwise_generator_permution_index),[-1, POI_NUM, 1])
        self.pointwise_permuation_mask = tf.expand_dims(self.pointwise_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
        self.pointwise_permuation_mask = tf.stop_gradient(self.pointwise_permuation_mask)
        # self.Q_network_output = self.out 

    
    #位置分布ListWise
    def _Listwise_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            listwise_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.listwise_cur_page_oam_output,self.listwise_intra_attr_output_each_poi = self._oam(listwise_cate_feature_embeddings_stopped)


            self.listwise_behavior_page_oam_output = tf.identity(self.listwise_cur_page_oam_output)
            for i in range(PAGE_NUM):
                listwise_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.listwise_tmp_behavior_input = tf.reshape(tf.gather(listwise_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_behavior_oam_output,_ = self._oam(self.listwise_tmp_behavior_input)
                if i == 0:
                    self.listwise_behavior_page_oam_output = tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)
                else:
                    self.listwise_behavior_page_oam_output = tf.concat([self.listwise_behavior_page_oam_output,tf.expand_dims(self.listwise_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            listwise_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.listwise_user_interest_emb = self._user_interest_target_attention(self.listwise_cur_page_oam_output,self.listwise_behavior_page_oam_output,listwise_mask_feature_stopped)
            
            #features['page_mask'] noindp

            self.listwise_poi_rep_emb = self.listwise_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.listwise_tmp_poi_input = tf.reshape(tf.gather(self.listwise_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_poi_output = self._intra_poi_self_attention(self.listwise_tmp_poi_input)
                if i == 0:
                    self.listwise_poi_rep_emb = tf.expand_dims(self.listwise_tmp_poi_output, axis=1)
                else:
                    self.listwise_poi_rep_emb = tf.concat([self.listwise_poi_rep_emb, tf.expand_dims(self.listwise_tmp_poi_output, axis=1)], axis=1)

            self.listwise_cur_page_oam_output = tf.tile(tf.expand_dims(self.listwise_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.listwise_user_interest_emb = tf.tile(tf.expand_dims(self.listwise_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            listwise_fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Listwise_MLP_A" + str(i)
                listwise_fc_out = tf.layers.dense(listwise_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                listwise_fc_out = tf.nn.swish(listwise_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            listwise_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            listwise_fc_input = tf.concat([self.listwise_cur_page_oam_output, self.listwise_user_interest_emb, self.listwise_poi_rep_emb, listwise_feat_predict_stopped,listwise_fc_out], axis=2)

            listwise_fc_out_ctr, listwise_fc_out_imp = tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "Listwise_MLP_B" + str(i)
                listwise_fc_out_ctr = tf.layers.dense(listwise_fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                listwise_fc_out_ctr = tf.nn.relu(listwise_fc_out_ctr)

            listwise_ctr_out = tf.layers.dense(listwise_fc_out_ctr, OUT_NUM, activation=None, name="listwise_final_out_ctr")

            if not self.train:
                listwise_ctr_out = tf.nn.sigmoid(listwise_ctr_out)

            self.listwise_generator_out = tf.reshape(listwise_ctr_out,[-1,POI_NUM]) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.listwise_generator_permution_index = tf.argsort(self.listwise_generator_out, axis=1, direction='DESCENDING') #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.listwise_generator_permution_index = tf.stop_gradient(self.listwise_generator_permution_index)

            self.listwise_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.listwise_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.listwise_generator_embedding = tf.expand_dims(self.listwise_generator_embedding, axis=1)  # [bs, 1, 5,7,8]
            self.listwise_generator_embedding = tf.stop_gradient(self.listwise_generator_embedding)

            self.listwise_generator_permution_index_expand = tf.expand_dims(self.listwise_generator_permution_index, axis=1)  # [bs, 1, 5]
            self.listwise_generator_permution_index_expand = tf.stop_gradient(self.listwise_generator_permution_index_expand)

            self.listwise_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.listwise_generator_permution_index),[-1, POI_NUM, 1])
            self.listwise_permuation_mask = tf.expand_dims(self.listwise_permuation_mask, axis=1)  # [bs,1,5,1]
            self.listwise_permuation_mask = tf.stop_gradient(self.listwise_permuation_mask)
            # self.Q_network_output = self.out 
    
    #生成式ListWise
    def _headlayers(self, top_tensor_list, position_tensor):
        """
        top_tensor_list: [batch_size, 50 embedding_size]]
        position_tensor: [batch_size , embedding_size]
        output: [batch_size, predict_len, 1]
        """
        position_tensor = tf.tile(tf.expand_dims(position_tensor, axis=1), [1, POI_NUM, 1])
        output = tf.concat([top_tensor_list, position_tensor], axis=2)
        with tf.variable_scope("head_layers", reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(output, 64, activation=tf.nn.relu, name="head_1")
            output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="head_2")
            output = tf.layers.dense(output, 1, activation=None, name="score")
        return output

    def _Generative_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.generative_cur_page_oam_output,self.generative_intra_attr_output_each_poi = self._oam(generative_cate_feature_embeddings_stopped)

            self.generative_behavior_page_oam_output = tf.identity(self.generative_cur_page_oam_output)
            for i in range(PAGE_NUM):
                generative_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.generative_tmp_behavior_input = tf.reshape(tf.gather(generative_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_behavior_oam_output,_ = self._oam(self.generative_tmp_behavior_input)
                if i == 0:
                    self.generative_behavior_page_oam_output = tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)
                else:
                    self.generative_behavior_page_oam_output = tf.concat([self.generative_behavior_page_oam_output,tf.expand_dims(self.generative_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            generative_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.generative_user_interest_emb = self._user_interest_target_attention(self.generative_cur_page_oam_output,self.generative_behavior_page_oam_output,generative_mask_feature_stopped)

            self.generative_poi_rep_emb = self.generative_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.generative_tmp_poi_input = tf.reshape(tf.gather(self.generative_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_poi_output = self._intra_poi_self_attention(self.generative_tmp_poi_input)
                if i == 0:
                    self.generative_poi_rep_emb = tf.expand_dims(self.generative_tmp_poi_output, axis=1)
                else:
                    self.generative_poi_rep_emb = tf.concat([self.generative_poi_rep_emb, tf.expand_dims(self.generative_tmp_poi_output, axis=1)], axis=1)

            self.generative_cur_page_oam_output = tf.tile(tf.expand_dims(self.generative_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.generative_user_interest_emb = tf.tile(tf.expand_dims(self.generative_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            generative_fc_out = tf.reshape(generative_cate_feature_embeddings_stopped,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Generative_MLP_A" + str(i)
                generative_fc_out = tf.layers.dense(generative_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                generative_fc_out = tf.nn.swish(generative_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            generative_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            generative_fc_input = tf.concat([self.generative_cur_page_oam_output, self.generative_user_interest_emb, self.generative_poi_rep_emb, generative_feat_predict_stopped,generative_fc_out], axis=2) # Batch_size * POI_NUM * FEAT_NUM
            generative_fc_input = tf.layers.dense(generative_fc_input, 64, activation=None, name="generative_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            batch_size      = tf.shape(generative_fc_input)[0]
            with tf.variable_scope("rnn_init_state", reuse=tf.AUTO_REUSE):
                state = tf.reduce_mean(generative_fc_input, axis=1)  # [B,64]
                state = tf.stop_gradient(state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                input_tensor = tf.zeros_like(state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64, name='gru_cell')  # 替换 tf.contrib.rnn.GRUCell
        
            if self.train:
                self.label = labels['ctr_label']
                self.generative_generator_loss = 0.0
                output_list = []
                label_list = []
                eval_weight_list = []
                generative_generator_out = []
                generative_generator_permution_index = []
                mask_tmp = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]
                eval_ones = tf.ones([batch_size, POI_NUM], dtype=tf.float32) # [B, 5]

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 5,1], head_layer最后一层不用激活函数
                    head_score_origin = tf.reshape(head_score, [-1, POI_NUM]) #[B, 5]
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - mask_tmp, -1.0e9), head_score_origin)) #[B,5]
                    # lable
                    select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1), [tf.shape(head_score)[0], 1]) #[B, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32) #[B, 5]

                    # sample_reward = realshow_list * sample_weight
                    label_mask = self.label * select_onehot #[B, 5], 只有曝光位置有reward, 其他为 0
                    # loss
                    tmp_loss = -1.0 * tf.reduce_sum(label_mask * tf.log(tf.clip_by_value(head_score, 1e-9, 1.0)), axis=-1) # [B]
                    tmp_loss = tf.reduce_mean(tmp_loss)
                    self.generative_generator_loss = self.generative_generator_loss + tmp_loss
                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
                    

                    output_list.append(head_score)
                    label_list.append(label_mask)
                    weight = mask_tmp * eval_ones
                    eval_weight_list.append(weight)

                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [B, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]
                    input_tensor = tf.stop_gradient(input_tensor)
            else:
                generative_generator_permution_index = []
                generative_generator_out = []
                mask_tmp = tf.ones([batch_size, PAGE_NUM], dtype=tf.float32) # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor) #[B, 50,1]
                    head_score = tf.reshape(head_score, [-1, POI_NUM])
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), head_score)) #[B,50]
   
                    select_index = tf.reshape(tf.argmax(head_score, axis=1, output_type=tf.int32), [-1, 1])  # [batch_size, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32)
                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [batch_size, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]

                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))
            


                

            self.generative_generator_out = tf.stack(generative_generator_out, axis=1) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.generative_generator_permution_index = tf.stack(generative_generator_permution_index, axis=1) # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.generative_generator_permution_index = tf.stop_gradient(self.generative_generator_permution_index)

            self.generative_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.generative_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.generative_generator_embedding = tf.expand_dims(self.generative_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.generative_generator_embedding = tf.stop_gradient(self.generative_generator_embedding)

            self.generative_generator_permution_index_expand = tf.expand_dims(self.generative_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
            self.generative_generator_permution_index_expand = tf.stop_gradient(self.generative_generator_permution_index_expand)

            self.generative_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.generative_generator_permution_index),[-1, POI_NUM, 1])
            self.generative_permuation_mask = tf.expand_dims(self.generative_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
            self.generative_permuation_mask = tf.stop_gradient(self.generative_permuation_mask)
            # self.Q_network_output = self.out

    #Submodularity Model
    def _Submodularity_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.submodularity_cur_page_oam_output,self.submodularity_intra_attr_output_each_poi = self._oam(submodularity_cate_feature_embeddings_stopped)

            self.submodularity_behavior_page_oam_output = tf.identity(self.submodularity_cur_page_oam_output)
            for i in range(PAGE_NUM):
                submodularity_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(self.behavior_cate_feature_embeddings)
                self.submodularity_tmp_behavior_input = tf.reshape(tf.gather(submodularity_behavior_cate_feature_embeddings_stopped,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.submodularity_tmp_behavior_oam_output,_ = self._oam(self.submodularity_tmp_behavior_input)
                if i == 0:
                    self.submodularity_behavior_page_oam_output = tf.expand_dims(self.submodularity_tmp_behavior_oam_output,axis=1)
                else:
                    self.submodularity_behavior_page_oam_output = tf.concat([self.submodularity_behavior_page_oam_output,tf.expand_dims(self.submodularity_tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            submodularity_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.submodularity_user_interest_emb = self._user_interest_target_attention(self.submodularity_cur_page_oam_output,self.submodularity_behavior_page_oam_output,submodularity_mask_feature_stopped)

            self.submodularity_poi_rep_emb = self.submodularity_intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.submodularity_tmp_poi_input = tf.reshape(tf.gather(self.submodularity_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.submodularity_tmp_poi_output = self._intra_poi_self_attention(self.submodularity_tmp_poi_input)
                if i == 0:
                    self.submodularity_poi_rep_emb = tf.expand_dims(self.submodularity_tmp_poi_output, axis=1)
                else:
                    self.submodularity_poi_rep_emb = tf.concat([self.submodularity_poi_rep_emb, tf.expand_dims(self.submodularity_tmp_poi_output, axis=1)], axis=1)

            self.submodularity_cur_page_oam_output = tf.tile(tf.expand_dims(self.submodularity_cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.submodularity_user_interest_emb = tf.tile(tf.expand_dims(self.submodularity_user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            submodularity_fc_out = tf.reshape(submodularity_cate_feature_embeddings_stopped,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Submodualrity_MLP_A" + str(i)
                submodularity_fc_out = tf.layers.dense(submodularity_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                submodularity_fc_out = tf.nn.swish(submodularity_fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            submodularity_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            submodularity_fc_input = tf.concat([self.submodularity_cur_page_oam_output, self.submodularity_user_interest_emb, self.submodularity_poi_rep_emb, submodularity_feat_predict_stopped,submodularity_fc_out], axis=2) # Batch_size * POI_NUM * FEAT_NUM
            submodularity_fc_input = tf.layers.dense(submodularity_fc_input, 64, activation=None, name="submodularity_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            submodularity_batch_size      = tf.shape(submodularity_fc_input)[0]
            with tf.variable_scope("submodularity_rnn_init_state", reuse=tf.AUTO_REUSE):
                submodularity_state = tf.reduce_mean(submodularity_fc_input, axis=1)  # [B,64]
                submodularity_state = tf.stop_gradient(submodularity_state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                submodularity_input_tensor = tf.zeros_like(submodularity_state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("submodularity_rnn", reuse=tf.AUTO_REUSE):
                submodularity_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64, name='submodularity_gru_cell')  # 替换 tf.contrib.rnn.GRUCell
        
            if self.train:
                self.label = labels['ctr_label']
                self.submodularity_generator_loss = 0.0
                submodularity_output_list = []
                submodularity_label_list = []
                submodularity_eval_weight_list = []
                submodularity_generator_out = []
                submodularity_generator_permution_index = []
                submodularity_mask_tmp = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32) # [B, 5]
                submodularity_eval_ones = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32) # [B, 5]

                #embedding distance
                submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
                feature_embedding_expanded = tf.expand_dims(submodularity_cate_feature_embeddings_stopped, axis=1)  # [bs, 1, 5, 7, 8]
                other_list_embedding = tf.stop_gradient(self.generator_permuation_embeddings)  # [bs, k, 5, 7, 8]
                # 计算 A 和 AA 在 5, 7, 8 维度上的 L2 范数
                feature_embedding_norm = tf.norm(feature_embedding_expanded, axis=[3, 4], keepdims=True)  # [bs, 1, 5, 1, 1]
                other_list_embedding_norm = tf.norm(other_list_embedding, axis=[3, 4], keepdims=True)  # [bs, k, 5, 1, 1]
                feature_embedding_norm = tf.squeeze(feature_embedding_norm, axis=[-1, -2])  # 形状变为 [1024, 3, 5]
                other_list_embedding_norm = tf.squeeze(other_list_embedding_norm, axis=[-1, -2])  # 形状变为 [1024, 3, 5]
                
                # 计算 A 和 AA 的点积
                similarity = tf.reduce_sum(feature_embedding_expanded * other_list_embedding, axis=[3, 4])  # [bs, k,5]
                similarity = tf.stop_gradient(similarity)
                # 计算余弦相似度
                cosine_similarity = similarity / (tf.clip_by_value(feature_embedding_norm * other_list_embedding_norm, 1e-4, float('inf')))  # [bs, k,5]
                cosine_similarity = tf.stop_gradient(cosine_similarity)
                # 对 k 维度取最大值，获取 A 中每个 item 与 AA 中 k 个候选的最大相似度
                coverage_score = tf.reduce_mean(cosine_similarity, axis=1)  # [bs, 5]
                coverage_score = tf.stop_gradient(coverage_score)
                

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    submodularity_output_tensor, submodularity_state = submodularity_gru_cell(submodularity_input_tensor, submodularity_state)
                    submodularity_head_score = self._headlayers(submodularity_fc_input, submodularity_output_tensor) #[B, 5,1], head_layer最后一层不用激活函数
                    submodularity_head_score_origin = tf.reshape(submodularity_head_score, [-1, POI_NUM]) #[B, 5]
                    submodularity_head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - submodularity_mask_tmp, -1.0e9), submodularity_head_score_origin)) #[B,5]
                    # lable
                    submodularity_select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1), [tf.shape(submodularity_head_score)[0], 1]) #[B, 1]
                    submodularity_select_onehot = tf.one_hot(tf.reshape(submodularity_select_index, [-1]), POI_NUM, dtype=tf.float32) #[B, 5]

                    # sample_reward = realshow_list * sample_weight
                    submodularity_label_mask = self.label * submodularity_select_onehot #[B, 5], 只有曝光位置有reward, 其他为 0
                    submodularity_coverage_score = coverage_score * submodularity_select_onehot

                    # loss
                    submodularity_tmp_loss = -1.0 * tf.reduce_sum(submodularity_label_mask * tf.log(tf.clip_by_value(submodularity_head_score, 1e-6, 1.0)), axis=-1) - (-1.0 * tf.reduce_sum(submodularity_coverage_score * tf.log(tf.clip_by_value(submodularity_head_score, 1e-6, 1.0)), axis=-1)) # [B]
                    submodularity_tmp_loss = tf.reduce_mean(submodularity_tmp_loss)
                    self.submodularity_generator_loss = self.submodularity_generator_loss + submodularity_tmp_loss
                    submodularity_generator_out.append(tf.reduce_max(submodularity_head_score, axis=1))
                    submodularity_generator_permution_index.append(tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32))
                    
                    submodularity_output_list.append(submodularity_head_score)
                    submodularity_label_list.append(submodularity_label_mask)
                    submodularity_weight = submodularity_mask_tmp * submodularity_eval_ones
                    submodularity_eval_weight_list.append(submodularity_weight)

                    submodularity_mask_tmp = submodularity_mask_tmp - submodularity_select_onehot
                    submodularity_input_tensor = tf.batch_gather(submodularity_fc_input, submodularity_select_index)  # [B, 1, embedding_size]
                    submodularity_input_tensor = tf.reshape(submodularity_input_tensor, [-1, 64])  # [batch_size, embedding_size]
                    submodularity_input_tensor = tf.stop_gradient(submodularity_input_tensor)
            else:
                submodularity_generator_permution_index = []
                submodularity_generator_out = []
                submodularity_mask_tmp = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32) # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    submodularity_output_tensor, submodularity_state = submodularity_gru_cell(submodularity_input_tensor, submodularity_state)
                    submodularity_head_score = self._headlayers(submodularity_fc_input, submodularity_output_tensor) #[B, 5,1]
                    submodularity_head_score = tf.reshape(submodularity_head_score, [-1, POI_NUM])
                    submodularity_head_score = tf.nn.softmax(tf.add(tf.multiply(1. - submodularity_mask_tmp, -1.0e9), submodularity_head_score)) #[B,5]
   
                    submodularity_select_index = tf.reshape(tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32), [-1, 1])  # [batch_size, 1]
                    submodularity_select_onehot = tf.one_hot(tf.reshape(submodularity_select_index, [-1]), POI_NUM, dtype=tf.float32)
                    submodularity_mask_tmp = submodularity_mask_tmp - submodularity_select_onehot
                    submodularity_input_tensor = tf.batch_gather(submodularity_fc_input, submodularity_select_index)  # [batch_size, 1, embedding_size]
                    submodularity_input_tensor = tf.reshape(submodularity_input_tensor, [-1, 64])  # [batch_size, embedding_size]

                    submodularity_generator_out.append(tf.reduce_max(submodularity_head_score, axis=1))
                    submodularity_generator_permution_index.append(tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32))
            

            self.submodularity_generator_out = tf.stack(submodularity_generator_out, axis=1) # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.submodularity_generator_permution_index = tf.stack(submodularity_generator_permution_index, axis=1) # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.submodularity_generator_permution_index = tf.stop_gradient(self.submodularity_generator_permution_index)

            self.submodularity_generator_embedding = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.submodularity_generator_permution_index),[-1,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.submodularity_generator_embedding = tf.expand_dims(self.submodularity_generator_embedding, axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.submodularity_generator_embedding = tf.stop_gradient(self.submodularity_generator_embedding)

            self.submodularity_generator_permution_index_expand = tf.expand_dims(self.submodularity_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
            self.submodularity_generator_permution_index_expand = tf.stop_gradient(self.submodularity_generator_permution_index_expand)

            self.submodularity_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.submodularity_generator_permution_index),[-1, POI_NUM, 1])
            self.submodularity_permuation_mask = tf.expand_dims(self.submodularity_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
            self.submodularity_permuation_mask = tf.stop_gradient(self.submodularity_permuation_mask)

    def _OCPM(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.cur_page_oam_output,self.intra_attr_output_each_poi = self._oam(self.cate_feature_embeddings)

            self.behavior_page_oam_output = tf.identity(self.cur_page_oam_output)
            for i in range(PAGE_NUM):
                self.tmp_behavior_input = tf.reshape(tf.gather(self.behavior_cate_feature_embeddings,list(range(i,i+1)),axis=1),[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_behavior_oam_output,_ = self._oam(self.tmp_behavior_input)
                if i == 0:
                    self.behavior_page_oam_output = tf.expand_dims(self.tmp_behavior_oam_output,axis=1)
                else:
                    self.behavior_page_oam_output = tf.concat([self.behavior_page_oam_output,tf.expand_dims(self.tmp_behavior_oam_output,axis=1)],axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            self.user_interest_emb = self._user_interest_target_attention(self.cur_page_oam_output,self.behavior_page_oam_output,features['page_mask'])

            self.poi_rep_emb = self.intra_attr_output_each_poi # -1 * poi * 8
            for i in range(POI_NUM):
                self.tmp_poi_input = tf.reshape(tf.gather(self.intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),[-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_poi_output = self._intra_poi_self_attention(self.tmp_poi_input)
                if i == 0:
                    self.poi_rep_emb = tf.expand_dims(self.tmp_poi_output, axis=1)
                else:
                    self.poi_rep_emb = tf.concat([self.poi_rep_emb, tf.expand_dims(self.tmp_poi_output, axis=1)], axis=1)

            self.cur_page_oam_output = tf.tile(tf.expand_dims(self.cur_page_oam_output,axis=1),[1,POI_NUM,1])
            self.user_interest_emb = tf.tile(tf.expand_dims(self.user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            fc_out = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)

           # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat([self.cur_page_oam_output, self.user_interest_emb, self.poi_rep_emb, self.feat_predict,fc_out], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None, name=dense_name)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_model_final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out,[-1,POI_NUM])
            self.Q_network_output = self.out

    def _OCPM_TOP_K(self, features, labels, mode, params,k_page,k_predict,k,mask):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        target_page_embedding = tf.reshape(k_page,[-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        target_page_predict_fea = tf.reshape(k_predict,[-1, POI_NUM, 1])

        with tf.name_scope('dnn_model'):
            cur_page_oam_output, target_intra_attr_output_each_poi = self._oam(target_page_embedding)

            behavior_page_oam_output = self.behavior_page_oam_output

            user_interest_emb = self._user_interest_target_attention_topK(cur_page_oam_output,behavior_page_oam_output,features['page_mask'],k)

            poi_rep_emb = target_intra_attr_output_each_poi # （batch * k） * poi * 8
            for i in range(POI_NUM):
                tmp_poi_input = tf.reshape(
                    tf.gather(target_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                tmp_poi_output = self._intra_poi_self_attention(tmp_poi_input)
                if i == 0:
                    poi_rep_emb = tf.expand_dims(tmp_poi_output, axis=1)
                else:
                    poi_rep_emb = tf.concat(
                        [poi_rep_emb, tf.expand_dims(tmp_poi_output, axis=1)], axis=1)

            cur_page_oam_output = tf.tile(tf.expand_dims(cur_page_oam_output, axis=1),[1, POI_NUM, 1])
            user_interest_emb = tf.tile(tf.expand_dims(user_interest_emb, axis=1), [1, POI_NUM, 1])

            fc_out = tf.reshape(target_page_embedding, [-1, POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out = tf.nn.swish(fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat(
                [cur_page_oam_output, user_interest_emb, poi_rep_emb, target_page_predict_fea, fc_out],axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input,[-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i],
                                             activation=None, name=dense_name,reuse=tf.AUTO_REUSE)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_topk_final_out_ctr",reuse=tf.AUTO_REUSE)
            ctr_out = tf.nn.sigmoid(ctr_out)

            topk_out = tf.reshape(ctr_out, [-1, k, POI_NUM])
            mask = tf.reshape(mask,[-1,k,POI_NUM])
            return topk_out * mask

    


    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss
            self.pointwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.pointwise_generator_out, weights=self.mask))
            self.listwise_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.listwise_generator_out, weights=self.mask))
            # self.submodularity_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.submodularity_generator_out, weights=self.mask))
            self.generator_loss = self.pointwise_generator_loss + self.listwise_generator_loss  + self.generative_generator_loss + self.submodularity_generator_loss
            self.evaluator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.out, weights=self.mask))
            self.loss = self.generator_loss + self.evaluator_loss
            # if USE_CONSTRATIVE_LOSS:
            #     self.loss -= CONSTRATIVE_LOSS_K * (tf.reduce_mean(self.topK_output,axis=[0,1,2]) - tf.reduce_mean(self.lastK_output,axis=[0,1,2]))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
    
    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer(),
                                            )
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.intra_attr_self_attention_weights = {
                'weight_Q':tf.get_variable(name='intra_attr_self_attention_weights_Q',shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1],CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                           initializer=tf.random_normal_initializer(0.0, glorot),
                                           dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_attr_self_attention_weights_MLP',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32)
            }

            self.inter_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='inter_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.intra_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_poi_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_poi_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_poi_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_poi_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        cate_fea = tf.reshape(features['cate_feature'],[-1,POI_NUM * FEATURE_CATE_NUM])
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], cate_fea),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        full_permuation_index = tf.reshape(features['full_permuation_index'],[PERMUATION_SIZE,POI_NUM])

        self.cate_feature_embeddings_for_permuation = tf.reshape(self.cate_feature_embeddings,[-1,POI_NUM,FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.full_permuation_index = tf.reshape(tf.tile(tf.expand_dims(full_permuation_index,axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0],1,1]),[-1,POI_NUM * PERMUATION_SIZE])
        self.full_permuation_embeddings = tf.reshape(tf.batch_gather(self.cate_feature_embeddings_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,FEATURE_CATE_NUM,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        
        # self.input_embedding = tf.concat(
        #     [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']
        self.feat_predict_for_permuation = tf.reshape(self.feat_predict,[-1,POI_NUM,1])

        self.full_permuation_feat_predict = tf.reshape(tf.batch_gather(self.feat_predict_for_permuation,self.full_permuation_index),[-1,PERMUATION_SIZE,POI_NUM,1])


        self.permuation_mask = tf.reshape(features['permuation_mask'],[-1,POI_NUM,1])
        self.full_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.full_permuation_index),[-1, PERMUATION_SIZE, POI_NUM, 1])


        behavior_cate_fea = tf.reshape(features['behavior_cate_feature'],[-1,PAGE_NUM * POI_NUM * FEATURE_CATE_NUM])
        self.behavior_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], behavior_cate_fea),
            [-1, PAGE_NUM , POI_NUM, FEATURE_CATE_NUM ,CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        self.behavior_input_embedding = tf.concat([self.behavior_cate_feature_embeddings], axis=3)
        self.behavior_feat_predict = features['behavior_dense_feature']

        return features

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''

        def format_log(tensors):
            log0 = "train info: step {}, loss={:.4f}, ctr_loss={:.4f}, " \
                   "ctr_out={:.4f},  avg_ctr_topk={:.4f},  avt_ctr_lastk={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["loss"],
                tensors["ctr_out"],tensors['avg_ctr_topk'],tensors['avt_ctr_lastk']
            )
            return log0

        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "ctr_out": ctr_out,
                                                        "avg_ctr_topk" : tf.reduce_mean(self.topK_output,axis=[0,1,2]),
                                                        "avt_ctr_lastk" : tf.reduce_mean(self.lastK_output,axis=[0,1,2]),
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        self._FGSM(features)
        #pointwise输出一个序列
        self._Pointwise_generator_model(features, labels, mode, params)
        #位置分布输出一个序列
        self._Listwise_generator_model(features, labels, mode, params)
        #GRN输出一个序列
        self._Generative_generator_model(features, labels, mode, params)
        
        self.generator_permuation_embeddings = tf.concat([self.pointwise_generator_embedding,self.listwise_generator_embedding,self.generative_generator_embedding],axis =1) # [bs, k, 5,7,8]
        self.generator_permuation_embeddings = tf.stop_gradient(self.generator_permuation_embeddings)

        self.pointwise_generator_out_expand = tf.expand_dims(self.pointwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.listwise_generator_out_expand = tf.expand_dims(self.listwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generative_generator_out_expand = tf.expand_dims(self.generative_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generator_permuation_predict_fea = tf.concat([self.pointwise_generator_out_expand,self.listwise_generator_out_expand,self.generative_generator_out_expand],axis =1) # [bs, k, 5]
        self.generator_permuation_predict_fea = tf.stop_gradient(self.generator_permuation_predict_fea)

        self.generator_permuation_index = tf.concat([self.pointwise_generator_permution_index_expand,self.listwise_generator_permution_index_expand,self.generative_generator_permution_index_expand],axis =1)
        self.generator_permuation_index = tf.stop_gradient(self.generator_permuation_index)

        self.generator_permuation_mask = tf.concat([self.pointwise_permuation_mask,self.listwise_permuation_mask,self.generative_permuation_mask],axis =1) # [bs, k, 5,1]
        self.generator_permuation_mask = tf.stop_gradient(self.generator_permuation_mask)

        #全面性模型
        self._Submodularity_generator_model(features, labels, mode, params)

        self.submodularity_generator_out_expand = tf.expand_dims(self.submodularity_generator_out, axis=1)  # [bs, 1, 5]
        self.submodularity_generator_permuation_embeddings = tf.concat([self.generator_permuation_embeddings,self.submodularity_generator_embedding],axis =1) # [bs, k, 5,7,8]
        self.submodularity_generator_permuation_embeddings = tf.stop_gradient(self.submodularity_generator_permuation_embeddings)

        self.submodularity_generator_permuation_predict_fea = tf.concat([self.generator_permuation_predict_fea, self.submodularity_generator_out_expand],axis =1) # [bs, k, 5]
        self.submodularity_generator_permuation_predict_fea = tf.stop_gradient(self.submodularity_generator_permuation_predict_fea)

        self.submodularity_generator_permuation_index = tf.concat([self.generator_permuation_index,self.submodularity_generator_permution_index_expand],axis =1) # [bs, k, 5]
        self.submodularity_generator_permuation_index = tf.stop_gradient(self.submodularity_generator_permuation_index)

        self.submodularity_generator_permuation_mask = tf.concat([self.generator_permuation_mask,self.submodularity_permuation_mask],axis =1) # [bs, k, 5,1]
        self.submodularity_generator_permuation_mask = tf.stop_gradient(self.submodularity_generator_permuation_mask)

        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.pointwise_generator_embedding, self.pointwise_generator_out_expand,1,self.pointwise_permuation_mask)
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.listwise_generator_embedding, self.listwise_generator_out_expand,1,self.listwise_permuation_mask)
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generative_generator_out_expand,1,self.generative_permuation_mask)
        self._OCPM(features, labels, mode, params)
        generator_num = 4
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generator_permuation_predict_fea,generator_num,self.generator_permuation_mask)
        self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.submodularity_generator_permuation_embeddings, self.submodularity_generator_permuation_predict_fea,generator_num,self.submodularity_generator_permuation_mask)
        self.lastK_output = self._OCPM_TOP_K(features, labels, mode, params, self.last_k_permuation_embeddings, self.last_k_permuation_predict_fea, TOP_K,self.last_k_permuation_mask)
        self.randK_output = self._OCPM_TOP_K(features, labels, mode, params, self.random_k_permuation_embeddings, self.random_k_permuation_predict_fea, TOP_K,self.random_k_permuation_mask)
        self.beamK_output = self._OCPM_TOP_K(features, labels, mode, params, self.beam_search_k_permuation_embeddings, self.beam_search_k_permuation_predict_fea, TOP_K,self.beam_search_k_permuation_mask)
        self.allK_output = self._OCPM_TOP_K(features, labels, mode, params, self.full_permuation_embeddings, self.full_permuation_feat_predict, PERMUATION_SIZE,self.full_permuation_mask)

        # 输出avg-ctr最高的序列
        self.avg_ctr_of_topK = tf.reduce_mean(self.topK_output,axis=2)
        max_ctr_index_in_topk = tf.nn.top_k(self.avg_ctr_of_topK,k=1).indices # batch * 1
        self.final_rerank_output_index = tf.batch_gather(self.submodularity_generator_permuation_index,max_ctr_index_in_topk) # bs,1,5
        # self.max_ctr_index = tf.batch_gather(self.top_k_permutation_indicces,self.max_ctr_index_in_topk) # batch * 1
        # full_permuation_index = tf.reshape(features['full_permuation_index'], [PERMUATION_SIZE, POI_NUM])
        # full_permuation_index = tf.tile(tf.expand_dims(full_permuation_index, axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0], 1, 1]) # batch * 120 * 5

        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op, training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out"),
                    'output_index':tf.identity(self.final_rerank_output_index,"out_index")
                    }
            else:
                ctr_out = self.out
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                           "ctr_topk": self.topK_output,
                           "ctr_lastk": self.lastK_output,
                           "ctr_randk": self.randK_output,
                           "ctr_all": self.allK_output,
                           "ctr_beamk": self.beamK_output,
                           'out_index':self.final_rerank_output_index
                         }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)


    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/config.py
````python
# coding: utf-8 -*-
import shutil
import time
import os

# result
RANDOM_SEED = 2022
BATCH_SIZE = 1024
IMP_LOSS_WEIGHT = 0.02
# basic config
EPOCH = 1
LEARNING_RATE = 0.001
DATA_MODE = 3 # 1:local train，2:local test, 3:docker evaluate
TRAIN_MODE = 3
MODEL_NAME = "MultiGE_submodularity"

USE_CONSTRATIVE_LOSS = True
CONSTRATIVE_LOSS_K = 0.01

# poi类别特征
FEATURE_CATE_NUM = 7 # v1r3:19
# dense特征
FEATURE_DENSE_NUM = 5  # v1:28 v1r2:79 v1r3:83
# 预估值特征
FEATURE_CXR_NUM = 3
# 环境特征
FEATURE_ENV_NUM = 2
# 自然poi
FEATURE_NATURE_POI = 25

# embedding_look_up维度
CATE_FEATURE_EMBEDDINGS_SHAPE = [1 << 22, 8]

# hash vector
HASH_VECTOR = [[0.4011729380078948, 0.9907933488085712, 0.889170658914898, 0.015553701343792192, 0.2269008585804, 0.889609750280199, 0.19280345796299014, 0.33362195188257815],
               [0.1174580997476552, 0.4475570392889796, 0.9925939893135071, 0.7296640075216434, 0.6436313332140967, 0.8568068597644793, 0.9018604021483339, 0.28774343258170776],
               [0.4374993384759095, 0.6807149381274915, 0.20502126763268802, 0.6968687323602859, 0.6449029002244834, 0.08732080642447282, 0.6119476780855001, 0.8616788453789646],
               [0.19344266090314144, 0.4268233179146762, 0.10951597767480326, 0.9867838283258178, 0.8340011944969644, 0.7992329879482085, 0.005303560724105649, 0.9662924610057512],
               [0.5824282763301396, 0.5090708710943849, 0.4462703076663568, 0.5482943153972023, 0.06782829736851825, 0.08907408658464577, 0.13400486343251583, 0.1848228429919272]]

POSITION_ENCODING = [[0.08790239841717873, 0.033267486152506076, 0.5495130189114207, 0.17419777583517537, 0.6838981992197484, 0.07935154925635501, 0.02665372302227631, 0.4181008411574786],
                     [0.02038159041970189, 0.9310485215006198, 0.723661313625571, 0.5110323516977285, 0.7812851438212606, 0.40722710713947474, 0.04646490014890503, 0.9565353323381218],
                     [0.17279361698259843, 0.12968275664201512, 0.3183103529758954, 0.6360591081256931, 0.1558507653689548, 0.5972802646455662, 0.4380619835390329, 0.6088094249662641],
                     [0.038262298606345335, 0.9999545626217287, 0.7113776275017341, 0.5434618368150265, 0.6853527957705402, 0.9662653254145415, 0.9641592716989676, 0.3443813983264],
                     [0.7903243938847678, 0.9952713339078417, 0.8741415264071601, 0.45665348276461737, 0.7693872696125916, 0.772509599868299, 0.2540369924156157, 0.24781240400239857]]


TIME_AWARE_WEIGHT = [[1/2,1/2,1/3,1/3,1/4]]
PERMUATION_SIZE = 120
TOP_K = 20


EXPOSE_RATE_FOR_BEAM_SEARCH = [[1.0,0.9,0.8,0.7,0.6]]

# N: Cut Number of POI For Train
POI_NUM = 5
FEATURE_NUM = 9
PAGE_NUM = 5
FEATURE_NUM_FOR_PAGE = 11

# 属性特征：KA AOR BRAND
FEATURE_ATTR_NUM = 3

# DELIVERY_FEAT
DELIVERY_FEAT_NUM = 4

# OUT NUM
OUT_NUM = 1

PLACE_HOLDER_NUM = 11
DENSE_FEAT_NUM = 439


# 网络结构参数
MODEL_PARAMS = {
    'INPUT_TENSOR_LAYERS_A': [60, 32, 20],
    'INPUT_TENSOR_LAYERS_B': [128, 32],
    'INPUT_TENSOR_LAYERS_C': [50, 20],
    'INPUT_TENSOR_LAYERS_D': [50, 20],
    'INPUT_TENSOR_LAYERS_E': [50, 20]
}
# A_INPUT_DIM = POI_NUM * (MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1] + 1)
MLP_INPUT_DIM = CATE_FEATURE_EMBEDDINGS_SHAPE[1] * 3 + 1 + MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][-1]
DIN_CONF = {}

# train data
# /users/lemonace/Downloads/tfrecord-rl-limit5-v1
if DATA_MODE == 1:
    TRAIN_FILE = ['/users/meituan_sxw/Downloads/part-r-00046']
    VALID_FILE = TRAIN_FILE
    PREDICT_FILE = VALID_FILE
    TEST_FILE = PREDICT_FILE
elif DATA_MODE == 2:
    TRAIN_FILE = ['/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/test_data/part-r-*']
    VALID_FILE = TRAIN_FILE
    TEST_FILE = VALID_FILE
elif DATA_MODE == 3:
    TRAIN_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/train_data/part-r-*"]
    VALID_FILE = ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-*"]
    TEST_FILE= ["/home/liushuchang/.jupyter/hailan_workspace/avito_v3_new/test_data/part-r-*"]
elif DATA_MODE == 4:
    DATA_FILE = "/home/hadoop-hmart-waimaiad/cephfs/data/yangfan129/train_data/tfrecord-multi-channel-v1/"
    #TRAIN_LIST = ["20211222", "20211223", "20211224", "20211225"]
    TRAIN_LIST = ["20220123"]
    VALID_LIST = ["20220124"]
    TRAIN_FILE = [DATA_FILE + x + "/part-r-*" for x in TRAIN_LIST]
    VALID_FILE = [DATA_FILE + x + "/part-r-0001*" for x in VALID_LIST]
    TEST_FILE = [DATA_FILE + x + "/part-r-00011" for x in TRAIN_LIST]

# 辅助脚本
MEAN_VAR_PATH_POI = "./avg_std/poi"
MEAN_VAR_PATH_DELIVERY = "./avg_std/delivery"
MODEL_SAVE_PATH = "./model/" + MODEL_NAME
MODEL_SAVE_PB_EPOCH_ON = False
MODEL_SAVE_PB_EPOCH_PATH = MODEL_SAVE_PATH + "_pbs"
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/data_input.py
````python
# -*- coding: utf-8 -*-
import tensorflow as tf
from config import *
import numpy as np
from tools import tick_tock,allPermutation


def generate_parse_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def generate_parse_valid_tfrecord_local_fn():
    def _parse_function(batch_examples):
        common_features, sequence_features = feature_parse_scheme()
        parsed_features = tf.parse_example(
            serialized=batch_examples,
            features=common_features
        )
        features = feature_product(parsed_features)
        labels = label_product(parsed_features)
        return features, labels

    return _parse_function


def feature_parse_scheme():
    label_len = POI_NUM * 2 + PAGE_NUM
    feature_len = POI_NUM * FEATURE_NUM + PAGE_NUM * POI_NUM * FEATURE_NUM_FOR_PAGE
    common_features = {
        "label": tf.FixedLenFeature([label_len], dtype=tf.float32),
        "feature": tf.FixedLenFeature([feature_len], dtype=tf.float32),
    }

    sequence_features = {}
    return common_features, sequence_features


def label_product(parsed_features):
    labels = parsed_features['label']

    labels_result = {
        # ctr_label
        'ctr_label': tf.gather(labels, list(range(0, POI_NUM)), axis=1),
        'mask': tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        'page_mask': tf.gather(labels, list(range(2 * POI_NUM, 2 * POI_NUM + PAGE_NUM)), axis=1),
    }
    return labels_result


def feature_product(parsed_features):
    feature_buffer = parsed_features['feature']
    labels = parsed_features['label']
    # 获取特征
    # FEATURE_CATE_NUM：品类相关特征
    # FEATURE_DENSE_NUM：连续值特征
    # FEATURE_CXR_NUM：模型预估值特征

    full_permuation_index = allPermutation(POI_NUM)

    # current page
    current_page_start = 0
    current_page_end = current_page_start + POI_NUM * FEATURE_NUM

    pre_page_start = current_page_end
    pre_page_end = pre_page_start + PAGE_NUM * POI_NUM * FEATURE_NUM_FOR_PAGE

    
    cur_page_features = tf.reshape(tf.gather(feature_buffer,list(range(current_page_start,current_page_end)),axis=1), [-1, POI_NUM, FEATURE_NUM])
    pre_page_features = tf.reshape(tf.gather(feature_buffer,list(range(pre_page_start,pre_page_end)),axis=1), [-1, PAGE_NUM, POI_NUM, FEATURE_NUM_FOR_PAGE])
    
    position_fea = tf.gather(cur_page_features, list(range(0, 1)), axis=2)
    adid_fea = tf.gather(cur_page_features, list(range(1, 2)), axis=2)
    obj_type_fea = tf.gather(cur_page_features, list(range(2, 3)), axis=2)
    hist_ctr_fea = tf.gather(cur_page_features, list(range(3, 4)), axis=2)
    locationid_fea = tf.gather(cur_page_features, list(range(4, 5)), axis=2)
    categoryid_fea = tf.gather(cur_page_features, list(range(5, 6)), axis=2)
    price_fea =  tf.gather(cur_page_features, list(range(6, 7)), axis=2)
    iscontext_fea = tf.gather(cur_page_features, list(range(7, 8)), axis=2)
    userid_fea = tf.gather(cur_page_features, list(range(8, 9)), axis=2)

    pre_position_fea = tf.gather(pre_page_features, list(range(0, 1)), axis=3)
    pre_adid_fea = tf.gather(pre_page_features, list(range(1, 2)), axis=3)
    pre_obj_type_fea = tf.gather(pre_page_features, list(range(2, 3)), axis=3)
    pre_hist_ctr_fea = tf.gather(pre_page_features, list(range(3, 4)), axis=3)
    pre_locationid_fea = tf.gather(pre_page_features, list(range(4, 5)), axis=3)
    pre_categoryid_fea = tf.gather(pre_page_features, list(range(5, 6)), axis=3)
    pre_price_fea = tf.gather(pre_page_features, list(range(6, 7)), axis=3)
    pre_iscontext_fea = tf.gather(pre_page_features, list(range(7, 8)), axis=3)
    pre_userid_fea = tf.gather(pre_page_features, list(range(8, 9)), axis=3)

    random_indices = np.random.randint(0,PERMUATION_SIZE,size=[1,TOP_K],dtype=np.int32)

    features_result = {
        
        'dense_feature': hist_ctr_fea,
        'permuation_mask' : tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        # 离散特征(品类)
        'cate_feature': tf.cast(tf.concat([position_fea, adid_fea, obj_type_fea, locationid_fea, iscontext_fea, categoryid_fea, userid_fea], axis=2), tf.int64),
        # ctr_label
        'ctr_label': tf.gather(labels, list(range(0, POI_NUM)), axis=1),
        'mask': tf.gather(labels, list(range(POI_NUM, 2 * POI_NUM)), axis=1),
        'page_mask':tf.gather(labels, list(range(2 * POI_NUM, 2 * POI_NUM + PAGE_NUM)), axis=1),
        'behavior_dense_feature': pre_hist_ctr_fea,
        # 离散特征(品类)
        'behavior_cate_feature': tf.cast(tf.concat([pre_position_fea, pre_adid_fea, pre_obj_type_fea, pre_locationid_fea,
                                                    pre_iscontext_fea, pre_categoryid_fea, pre_userid_fea],axis=3), tf.int64),

        'hash_vector' : tf.constant(HASH_VECTOR,tf.float32),
        'position_encoding': tf.constant(POSITION_ENCODING, tf.float32),
        'disantce_weight' : tf.constant(TIME_AWARE_WEIGHT,tf.float32),
        'random_indices' : tf.constant(random_indices,tf.int32)
    }
    features_result['full_permuation_index'] = tf.constant(full_permuation_index,tf.int32)
    return features_result


# num_parallel 表示cpu的核数,用于控制 map的并行度
def input_fn_maker(file_names, is_train, batch_size, epoch=None, num_parallel=4):
    def input_fn():
        _parse_fn = generate_parse_tfrecord_local_fn() if is_train else generate_parse_valid_tfrecord_local_fn()
        files = tf.data.Dataset.list_files(file_names)
        # print(files)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4 * 10))
        dataset = dataset.prefetch(buffer_size=batch_size * 10)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_fn, num_parallel_calls=num_parallel)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


# 从hive表统计得到均值和方差文件
def get_normalization_parameter(mean_var_path):
    with tf.gfile.Open(mean_var_path) as f:
        fea_mean = f.readline().strip().split(' ')
        fea_var = f.readline().strip().split(' ')
        cont_fea_mean = list(map(float, fea_mean))
        cont_fea_var = list(map(float, fea_var))
    f.close()
    return cont_fea_mean, cont_fea_var


def get_bias_weight_parameter(bias_weight_path):
    with tf.gfile.Open(bias_weight_path) as f2:
        fea_mean = f2.readline().strip().split('\t')
        cont_fea_mean = list(map(float, fea_mean))
    f2.close()
    return cont_fea_mean



if __name__ == '__main__':
    train_file = TRAIN_FILE
    # train_file = ["/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yangfan129/train_data/avito_v1_new/avito_v1_new/train_data/part-r-00000"]
    train_input_fn = input_fn_maker(train_file, is_train=True, batch_size=16, epoch=1)
    features, labels = train_input_fn()

    sess = tf.Session()
    try:
        with tick_tock("DATA_INPUT") as _:
            features_np, labels_np = sess.run([features, labels])

        print("*" * 100, "features_np")
        for key in features_np:
            print("=" * 50, key, np.shape(features_np[key]))
            print(features_np[key])


        print("*" * 100, "labels_np")
        for key in labels_np:
            print("=" * 50, key, np.shape(labels_np[key]))
            print(labels_np[key])

    except Exception as e:
        print(e)
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/layers.py
````python
import tensorflow as tf

def DIN(seq, seq_len, target, conf, scope="DIN"):
    # seq BATCH_SIZE * SEQ_LEN * FEAT_NUM : N * M * H
    # target BATCH_SIZE * FEAT_NUM : N * H
    # return : BATCH_SIZE * H

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        seq_shape = tf.shape(seq)
        target = tf.tile(target, [1, seq_shape[1], 1])

        input = tf.concat([seq, target, seq - target, seq * target], axis=-1)

        layers = conf.get("layers", [64, 32])
        for layer in layers:
            input = tf.layers.dense(input, layer, activation=tf.nn.sigmoid, name="att_"+str(layer))

        input = tf.squeeze(tf.layers.dense(input, 1, activation=None, name="att_final"), axis=-1) # N * M

        # Mask
        seq_mask = tf.squeeze(tf.sequence_mask(seq_len, seq_shape[1]))
        # seq_mask = tf.Print(seq_mask, [seq_mask], message="seq_mask", summarize=100)
        padding = tf.ones_like(input) * (-2 ** 32 + 1)
        attention = tf.nn.softmax(tf.where(seq_mask, input, padding), axis=-1) # N * M
        # attention = tf.Print(attention, [attention], message="attention", summarize=100)
        attention = tf.tile(tf.expand_dims(attention, axis=2), [1, 1, seq_shape[2]])
        output = tf.reduce_sum(tf.transpose(attention * seq, [0, 2, 1]), axis=1)
    return output
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/main.py
````python
# -*- coding: utf-8 -*-
from config import *
from model import *
from sklearn import metrics
import numpy as np

def create_estimator():
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        model_dir=MODEL_SAVE_PATH,
        keep_checkpoint_max=2,
        log_step_count_steps=1000,
        session_config=session_config)
    nn_model = DNN()
    estimator = tf.estimator.Estimator(model_fn=nn_model.model_fn_estimator, config=config)
    return estimator, nn_model


def save_model_pb_with_estimator(estimator, params, export_dir_base):
    estimator._params['save_model'] = params['save_model']

    def _serving_input_receiver_fn():
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature
        receiver_tensors = {
            # ctr cvr gmv预估值 && bid
            'screen_predict_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_CXR_NUM],
                                                name='screen_predict_feature'),
            # dense 特征 (价格，评分)
            'screen_dense_feature': tf.placeholder(tf.float32, [None, POI_NUM, FEATURE_DENSE_NUM],
                                                name='screen_dense_feature'),
            # 离散特征(品类)
            'screen_cate_feature': tf.placeholder(tf.int64, [None, POI_NUM, FEATURE_CATE_NUM],
                                                name='screen_cate_feature'),
            # 环境特征（是否有铂金）
            'dense_feature': tf.placeholder(tf.float32, [None, DENSE_FEAT_NUM],
                                                name='dense_feature')
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)

    export_dir = estimator.export_saved_model(export_dir_base=export_dir_base,
                                              serving_input_receiver_fn=_serving_input_receiver_fn)
    estimator._params.pop('save_model')
    return export_dir.decode()


#def dcg_at_k(scores, k):
#    """
#    计算 DCG 值 (Discounted Cumulative Gain)
#    scores: 实际结果的评分
#    k: 截止位置
#    """
#    scores = np.asfarray(scores)[:k]  # 取前 k 个结果
#    if scores.size == 0:
#        return 0.0
#    # 使用公式：sum((2^relevance - 1) / log2(rank + 1))
#    return np.sum((2**scores - 1) / np.log2(np.arange(1, scores.size + 1) + 1))

#def ndcg_at_k(actual, predicted, k):
#    """
#    计算 NDCG 值 (Normalized Discounted Cumulative Gain)
#    actual: 实际的评分或标签
#    predicted: 预测结果的排序索引
#    k: 截止位置
#    """
#    ideal_sorted = sorted(actual, reverse=True)  # 理想的排序（从高到低）
#    dcg = dcg_at_k([actual[i] for i in predicted], k)  # 使用预测排序计算 DCG
#    idcg = dcg_at_k(ideal_sorted, k)  # 使用理想排序计算 IDCG
#    return dcg / idcg if idcg > 0 else 0.0

def dcg_at_k(scores, k):
    """
    计算给定排序情况下的 DCG 值
    :param scores: 真实排序后的标签值
    :param k: 计算前 k 个位置的 DCG
    :return: DCG 值
    """
    scores = np.asfarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2**scores - 1) / np.log2(np.arange(1, scores.size + 1) + 1))

def ndcg_at_k(prediction, label, k):
    """
    计算 NDCG@k
    :param prediction: 预测分数，形状为[batch_size, k]
    :param label: 理想排序后的标签，形状为[batch_size, k]
    :param k: 计算前 k 个位置的 NDCG
    :return: NDCG@k 值
    """
    batch_size = prediction.shape[0]
    ndcg_scores = np.zeros(batch_size)

    for i in range(batch_size):
        pred_scores = prediction[i]
        true_labels = label[i]

        # 按照预测的得分排序
        pred_ranks = np.argsort(-pred_scores)  # 降序排列

        # 计算 DCG
        dcg = dcg_at_k(true_labels[pred_ranks], k)

        # 计算 IDCG (理想的排序）
        idcg = dcg_at_k(true_labels, k)

        # 计算 NDCG
        if idcg > 0:
            ndcg_scores[i] = dcg / idcg
        else:
            ndcg_scores[i] = 0.0

    return np.mean(ndcg_scores)

def calculate_result(result_generator):

    y_ctr, pred_ctr, ctr = [], [], []
    topk_ctr = []
    beamK_ctr = []
    all_topk_ctr = []
    randomk_ctr = []
    best_topk_ctr = []
    best_beamK_ctr = []
    best_all_topk_ctr = []
    best_randomk_ctr = []
    for result in result_generator:
        cxr_feature = result['cxr_feature']
        mask = result['mask']
        #out_index = result['out_index']
        #print(out_index)
        ctr_out = result['ctr_out']
        ndcg_label = np.arange(POI_NUM - 1, -1, -1, dtype=np.float32)
        ndcg_label = np.expand_dims(ndcg_label, axis=0)
        ndcg_label = np.tile(ndcg_label, (1024, 1))
        ndcg_value = ndcg_at_k(ctr_out,ndcg_label,k=POI_NUM)
        
        #print("ctr_out:",ctr_out)
        #print("ctr_out.shape:",ctr_out.shape)
        #before = result['ctr_out'].reshape(-1).tolist()
        #print("before:",before)
        # ctr_label
        idx = np.where(mask.reshape(-1) == 1)
        listctr = result['ctr_out'].reshape(-1)[idx].tolist()
        #print("listctr:",listctr)
        #print("listctr:",len(listctr))
        y_ctr += result['ctr_label'].reshape(-1)[idx].tolist()
        pred_ctr += result['ctr_out'].reshape(-1)[idx].tolist()
        ctr += cxr_feature[:, :, 0].reshape(-1)[idx].tolist()

        tmp_ctr_topk = np.mean(result['ctr_topk'],axis=2).reshape(-1)
        tmp_ctr_randk = np.mean(result['ctr_randk'],axis=2).reshape(-1)
        tmp_ctr_all = np.mean(result['ctr_all'],axis=2).reshape(-1)
        tmp_ctr_all_topk = tmp_ctr_all[np.argsort(tmp_ctr_all * -1)].reshape(-1)
        tmp_ctr_beamk = np.mean(result['ctr_beamk'],axis=2).reshape(-1)

        topk_ctr += tmp_ctr_topk.tolist()
        #print("tmp_ctr_topk.tolist():",tmp_ctr_topk.tolist())
        beamK_ctr += tmp_ctr_beamk.tolist()
        randomk_ctr += tmp_ctr_randk.tolist()
        all_topk_ctr += tmp_ctr_all_topk.tolist()

        best_topk_ctr.append(np.max(tmp_ctr_topk))
        #print("np.max(tmp_ctr_topk)",np.max(tmp_ctr_topk))
        best_beamK_ctr.append(np.max(tmp_ctr_beamk))
        best_all_topk_ctr.append(np.max(tmp_ctr_all_topk))
        best_randomk_ctr.append(np.max(tmp_ctr_randk))

    #print("best_topk_ctr:",best_topk_ctr)
    #print("topk_ctr:",topk_ctr)
    #print("pred_ctr:",pred_ctr)
    #print("shape best_topk_ctr:",len(best_topk_ctr))
    #print("shape topk_ctr:",len(topk_ctr))
    #print("shape pred_ctr:",len(pred_ctr))
   
    #evaluator_out_argsort = np.argsort(pred_ctr)[::-1]
    #ndcg_value = ndcg_at_k(ndcg_label, evaluator_out_argsort, POI_NUM)
    ndcg_value = np.mean(ndcg_value)
     

    ctr_auc, ctr_ndcg, ctr_auc_jp, ctr_cb, ctr_cb_jp = metrics.roc_auc_score(y_ctr, pred_ctr), ndcg_value, metrics.roc_auc_score(y_ctr, ctr), np.sum(pred_ctr) / np.sum(y_ctr),  np.sum(ctr) / np.sum(y_ctr)
    print("ctr_auc:{}, ctr_ndcg:{}, ctr_auc_jp:{}, ctr_cb:{}, ctr_cb_jp:{}".format(ctr_auc, ctr_ndcg, ctr_auc_jp, ctr_cb, ctr_cb_jp))
    print("topk_ctr:{}, beamK_ctr:{}, randomk_ctr:{}, all_topk_ctr:{}".format(np.mean(topk_ctr), np.mean(beamK_ctr), np.mean(randomk_ctr), np.mean(all_topk_ctr)))
    print("best_topk_ctr:{}, best_beamK_ctr:{}, best_randomk_ctr:{}, best_all_topk_ctr:{}".format(np.mean(best_topk_ctr), np.mean(best_beamK_ctr), np.mean(best_randomk_ctr), np.mean(best_all_topk_ctr)))

if __name__ == '__main__':

    estimator, nn_model = create_estimator()

    with tick_tock("DATA_INPUT") as _:
        valid_input_fn = input_fn_maker(VALID_FILE, False, batch_size=1024, epoch=1)
        test_input_fn = input_fn_maker(TEST_FILE, False, batch_size=1024, epoch=1)

    if TRAIN_MODE == 1:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                if MODEL_SAVE_PB_EPOCH_ON:
                    export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                              export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
                    ep_insert_index = i * len(TRAIN_FILE) + idx
                    target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    while os.path.exists(target_dir):
                        target_dir = export_dir + "/../ep" + str(ep_insert_index)
                    shutil.move(export_dir, target_dir)
                    print(time.strftime("%m-%d %H:%M:%S ",
                                        time.localtime(time.time())) + "export model PB: " + target_dir)
                #with tick_tock("PREDICT") as _:
                    #result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    #calculate_result(result_generator)

    elif TRAIN_MODE == 2:
        with tick_tock("PREDICT") as _:
            result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
            calculate_result(result_generator)

    elif TRAIN_MODE == 3:
        for i in range(EPOCH):
            for idx, data in enumerate(TRAIN_FILE):
                with tick_tock("DATA_INPUT") as _:
                    train_input_fn = input_fn_maker([data], True, batch_size=BATCH_SIZE, epoch=1)
                with tick_tock("TRAIN") as _:
                    estimator.train(train_input_fn)
                with tick_tock("PREDICT") as _:
                    result_generator = estimator.predict(input_fn=valid_input_fn, yield_single_examples=False)
                    print("valid_data")
                    calculate_result(result_generator)
                    #result_generator = estimator.predict(input_fn=test_input_fn, yield_single_examples=False)
                    print("train_data")
                    #calculate_result(result_generator)
                    # save pb

    
    elif TRAIN_MODE == 4:
        export_dir = save_model_pb_with_estimator(estimator, params={'save_model': 'listwise'},
                                                    export_dir_base=MODEL_SAVE_PB_EPOCH_PATH)
        ep_insert_index = 0
        target_dir = export_dir + "/../ep" + str(ep_insert_index)
        while os.path.exists(target_dir):
            target_dir = export_dir + "/../ep" + str(ep_insert_index)
        shutil.move(export_dir, target_dir)
        print(time.strftime("%m-%d %H:%M:%S ",
                            time.localtime(time.time())) + "export model PB: " + target_dir)
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/model.py
````python
# coding: utf-8 -*-
from data_input import *
from tools import *
from config import *
from util import *
import numpy as np
import tensorflow as tf
from layers import *
import os
from collections import namedtuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DNN:
    def __init__(self):
        pass

    # env_feature = > dense_feature
    # cxr_feature = > screen_predict_feature
    # cat_feature = > screen_cate_feature
    # dense_feature = > screen_dense_feature

    def inter_attr_self_attention(self, input):

        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.inter_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (
                               tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (
                               tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (
                               tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
                           self.inter_attr_self_attention_weights['weight_MLP'])

        return output

    def intra_attr_self_attention(self, input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_attr_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q,
                               (tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K,
                               (tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V,
                               (tf.shape(input)[0], POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        return output

    def _oam(self, input):

        self.intra_attr_output = tf.identity(input)
        intra_attr_output_each_poi = tf.identity(input)
        for i in range(FEATURE_CATE_NUM):
            self.tmp_intra_attr_input = tf.reshape(tf.gather(input, list(range(i, i + 1)), axis=2),
                                                   [-1, POI_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_intra_attr_output = self.intra_attr_self_attention(self.tmp_intra_attr_input)
            self.tmp_intra_attr_output_mlp = tf.matmul(self.tmp_intra_attr_input + self.tmp_intra_attr_output,
                                                       self.intra_attr_self_attention_weights['weight_MLP'])
            if i == 0:
                self.intra_attr_output = tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)
                intra_attr_output_each_poi = tf.expand_dims(
                    tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)
            else:
                self.intra_attr_output = tf.concat(
                    [self.intra_attr_output, tf.expand_dims(self.tmp_intra_attr_output_mlp, axis=1)], axis=1)
                intra_attr_output_each_poi = tf.concat([intra_attr_output_each_poi, tf.expand_dims(
                    tf.reshape(self.tmp_intra_attr_output, [-1, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]]), axis=2)],
                                                       axis=2)

        inter_attr_output = self.inter_attr_self_attention(self.intra_attr_output)

        # inter_attr_output: -1 * 8
        # intra_attr_output_each_poi : -1 * POI_NUM * feanum * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]
        return inter_attr_output, intra_attr_output_each_poi

    def _user_interest_target_attention(self, target, behavior, mask):
        target = tf.tile(tf.expand_dims(target, axis=1), [1, tf.shape(behavior)[1], 1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights, axis=1)  # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights, axis=2) * behavior, axis=1)
        return user_interest  # batch * 8

    def _user_interest_target_attention_topK(self, target, behavior, mask, k):
        target = tf.tile(tf.expand_dims(target, axis=1), [1, tf.shape(behavior)[1], 1])
        behavior = tf.tile(behavior, [k, 1, 1])
        mask = tf.tile(mask, [k, 1])
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        weights = tf.reduce_sum(tf.multiply(target, behavior), axis=2)
        weights = tf.where(tf.equal(mask, 1.0), weights, paddings)
        weights = tf.nn.softmax(weights, axis=1)  # batch * 5
        #
        # weights = tf.Print(weights, [weights],'weights:', summarize=100)
        # tf.identity(weights)

        user_interest = tf.reduce_sum(tf.expand_dims(weights, axis=2) * behavior, axis=1)
        return user_interest  # batch * 8

    def _intra_poi_self_attention(self, input):
        encoder_Q = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_Q'])  # (batch * 5) * 8
        encoder_K = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_K'])  # (batch * 5) * 8
        encoder_V = tf.matmul(tf.reshape(input, (-1, CATE_FEATURE_EMBEDDINGS_SHAPE[1])),
                              self.intra_poi_self_attention_weights['weight_V'])  # (batch * 5) * 8

        encoder_Q = tf.reshape(encoder_Q, (
        tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_K = tf.reshape(encoder_K, (
        tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4
        encoder_V = tf.reshape(encoder_V, (
        tf.shape(input)[0], FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))  # batch * 5 * 5

        attention_map = attention_map / 8
        attention_map = tf.nn.softmax(attention_map)  # batch * 5 * 5

        output = tf.reshape(tf.matmul(attention_map, encoder_V),
                            (-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]))  # batch * 5 * 4

        output = tf.matmul(tf.reshape(input, [-1, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[1]]) + output,
                           self.intra_poi_self_attention_weights['weight_MLP'])

        return output

    def positional_encoding(self, zero_pad=True, scale=True):

        N = 1
        T = 5
        num_units = 8

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # batch * 8

        position_enc = np.array(
            [[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs

    def _PPML(self, features):
        position_encoding = features['position_encoding']
        position_encoding = tf.expand_dims(tf.expand_dims(position_encoding, axis=0), axis=0)
        self.ppml_page = tf.identity(self.behavior_cate_feature_embeddings)
        self.ppml_permuation = tf.identity(self.behavior_cate_feature_embeddings)
        # batch * 5 * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_behavior_input_embedding = tf.reshape(
                tf.gather(self.behavior_cate_feature_embeddings, list(range(i, i + 1)), axis=3),
                [-1, PAGE_NUM, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_behavior_input_embedding = tf.reduce_mean(self.tmp_behavior_input_embedding * position_encoding,
                                                               axis=2)  # batch * page * emb
            if i == 0:
                self.ppml_page = tf.expand_dims(self.tmp_behavior_input_embedding, axis=2)
            else:
                self.ppml_page = tf.concat([self.ppml_page, tf.expand_dims(self.tmp_behavior_input_embedding, axis=2)],
                                           axis=2)
        self.ppml_page = tf.reduce_mean(self.ppml_page, axis=2)  # # batch * page * emb
        # batch * pe * 5 * 7 * 8
        for i in range(FEATURE_CATE_NUM):
            self.tmp_permuation_input_embedding = tf.reshape(
                tf.gather(self.full_permuation_embeddings, list(range(i, i + 1)), axis=3),
                [-1, PERMUATION_SIZE, POI_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.tmp_permuation_input_embedding = tf.reduce_mean(
                self.tmp_permuation_input_embedding * position_encoding, axis=2)  # batch * page * emb
            if i == 0:
                self.ppml_permuation = tf.expand_dims(self.tmp_permuation_input_embedding, axis=2)
            else:
                self.ppml_permuation = tf.concat(
                    [self.ppml_permuation, tf.expand_dims(self.tmp_permuation_input_embedding, axis=2)], axis=2)
        self.ppml_permuation = tf.reduce_mean(self.ppml_permuation, axis=2)  # # batch * permuation * emb

    def _SimHash_TopOne(self, features):
        page_hash_vector = tf.expand_dims(features['hash_vector'], axis=0)  # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector)  # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature, axis=1),
                                           [1, PERMUATION_SIZE, 1, 1])  # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'], axis=0),
                                                axis=0)  # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation, axis=2), [1, 1, PAGE_NUM, 1])
        self.permuation_hash_signature = tf.sign(
            self.ppml_permuation * permuation_hash_vector)  # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),
                                         axis=3)  # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(
            tf.expand_dims(features['disantce_weight'], axis=0) * hamming_distance, axis=2)  # batch * permution
        top_permutation_indicces = tf.nn.top_k(weighted_hamming_distance * -1, k=1).indices  # batch * k
        last_permutation_indicces = tf.nn.top_k(weighted_hamming_distance, k=1).indices
        return top_permutation_indicces, last_permutation_indicces

    def _SimHash(self, features):  # yhl:120 select topk(20)
        page_hash_vector = tf.expand_dims(features['hash_vector'], axis=0)  # batch * page * emb
        self.page_hash_signature = tf.sign(self.ppml_page * page_hash_vector)  # batch * page * emb
        self.page_hash_signature = tf.tile(tf.expand_dims(self.page_hash_signature, axis=1),
                                           [1, PERMUATION_SIZE, 1, 1])  # # batch * permution * page * emb
        permuation_hash_vector = tf.expand_dims(tf.expand_dims(features['hash_vector'], axis=0),
                                                axis=0)  # batch * permution * page * emb
        self.ppml_permuation = tf.tile(tf.expand_dims(self.ppml_permuation, axis=2), [1, 1, PAGE_NUM, 1])
        self.permuation_hash_signature = tf.sign(
            self.ppml_permuation * permuation_hash_vector)  # batch * permution * page * emb

        hamming_distance = tf.reduce_sum(tf.math.abs(self.permuation_hash_signature - self.page_hash_signature),
                                         axis=3)  # batch * permution * page
        weighted_hamming_distance = tf.reduce_sum(
            tf.expand_dims(features['disantce_weight'], axis=0) * hamming_distance, axis=2)  # batch * permution
        # # select top40
        # top_30_permutation_indicces = tf.nn.top_k(weighted_hamming_distance  * -1,k=TOP_K + 10).indices # batch * k
        top_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance * -1, k=TOP_K).indices  # batch * k
        last_k_permutation_indicces = tf.nn.top_k(weighted_hamming_distance, k=TOP_K).indices
        return top_k_permutation_indicces, last_k_permutation_indicces

    # 简化处理，使用全排列精排累积ctr的topK
    def BeamSearch(self):
        expose_rate = tf.expand_dims(tf.constant(EXPOSE_RATE_FOR_BEAM_SEARCH, tf.float32), axis=0)
        full_permuation_feat_predict = tf.reshape(self.full_permuation_feat_predict,
                                                  [-1, PERMUATION_SIZE, POI_NUM]) * expose_rate
        cul_ctr = tf.reduce_sum(full_permuation_feat_predict, axis=2)  # batch * permutation
        top_k_permutation_indicces = tf.nn.top_k(cul_ctr, k=TOP_K).indices  # batch * k
        return top_k_permutation_indicces

    def _FGSM(self, features):
        # get behavior and permuation emb
        self._PPML(features)
        self.top_k_permutation_indicces, last_k_permutation_indicces = self._SimHash(features)
        random_k_permutation_indicces = tf.tile(features['random_indices'],
                                                [tf.shape(self.top_k_permutation_indicces)[0], 1])
        beam_k_permutation_indicces = self.BeamSearch()

        # top k
        self.top_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,
                                                           self.top_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        # 在这个地方加scm,生成序列跟其他序列的距离
        self.top_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,
                                                            self.top_k_permutation_indicces)  # # batch * k * 5 * 1
        self.top_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,
                                                     self.top_k_permutation_indicces)  # # batch * k * 5 * 1
        self.top_k_permuation_embeddings = tf.stop_gradient(self.top_k_permuation_embeddings)
        self.top_k_permuation_predict_fea = tf.stop_gradient(self.top_k_permuation_predict_fea)
        self.top_k_permuation_mask = tf.stop_gradient(self.top_k_permuation_mask)

        # lask k
        self.last_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,
                                                            last_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.last_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,
                                                             last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,
                                                      last_k_permutation_indicces)  # # batch * k * 5 * 1
        self.last_k_permuation_embeddings = tf.stop_gradient(self.last_k_permuation_embeddings)
        self.last_k_permuation_predict_fea = tf.stop_gradient(self.last_k_permuation_predict_fea)
        self.last_k_permuation_mask = tf.stop_gradient(self.last_k_permuation_mask)

        # random k
        self.random_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,
                                                              random_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.random_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,
                                                               random_k_permutation_indicces)  # # batch * k * 5 * 1
        self.random_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,
                                                        random_k_permutation_indicces)  # # batch * k * 5 * 1

        # beam search k
        self.beam_search_k_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,
                                                                   beam_k_permutation_indicces)  # batch * k * 5 * 7 * 8
        self.beam_search_k_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,
                                                                    beam_k_permutation_indicces)  # # batch * k * 5 * 1
        self.beam_search_k_permuation_mask = tf.batch_gather(self.full_permuation_mask,
                                                             beam_k_permutation_indicces)  # # batch * k * 5 * 1

    # 单点分布Pointwise(indp)
    def _Pointwise_generator_model(self, features, labels, mode, params):
        # get behavior and permuation emb
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)
        self._PPML(features)
        self.top_permutation_indicces, last_permutation_indicces = self._SimHash_TopOne(features)

        # top k
        self.top_permuation_embeddings = tf.batch_gather(self.full_permuation_embeddings,
                                                         self.top_permutation_indicces)  # [bs, 1, 5, 7, 8]
        self.top_permuation_predict_fea = tf.batch_gather(self.full_permuation_feat_predict,
                                                          self.top_permutation_indicces)  # # batch * 1 * 5 * 1
        self.top_permuation_mask = tf.batch_gather(self.full_permuation_mask,
                                                   self.top_permutation_indicces)  # # batch * k * 5 * 1
        target_page_predict_fea = tf.reshape(self.top_permuation_predict_fea, [-1, POI_NUM, 1])

        self.top_permuation_embeddings = tf.stop_gradient(self.top_permuation_embeddings)
        self.top_permuation_predict_fea = tf.stop_gradient(self.top_permuation_predict_fea)
        self.top_permuation_mask = tf.stop_gradient(self.top_permuation_mask)

        pointwise_ctr_out = tf.layers.dense(target_page_predict_fea, OUT_NUM, activation=None,
                                            name="pointwise_final_out_ctr", reuse=tf.AUTO_REUSE)

        if not self.train:
            pointwise_ctr_out = tf.nn.sigmoid(pointwise_ctr_out)

        self.pointwise_generator_out = tf.reshape(pointwise_ctr_out, [-1, POI_NUM])  # bs*5
        # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
        # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
        # 使用 tf.argsort 来获取排序后的索引
        self.pointwise_generator_permution_index = tf.argsort(self.pointwise_generator_out, axis=1,
                                                              direction='DESCENDING')  # 第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
        self.pointwise_generator_permution_index = tf.stop_gradient(self.pointwise_generator_permution_index)

        self.pointwise_generator_embedding = tf.reshape(
            tf.batch_gather(self.cate_feature_embeddings_for_permuation, self.pointwise_generator_permution_index),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        self.pointwise_generator_embedding = tf.expand_dims(self.pointwise_generator_embedding,
                                                            axis=1)  # 形状变为 [bs, 1, 5,7,8]
        self.pointwise_generator_embedding = tf.stop_gradient(self.pointwise_generator_embedding)

        self.pointwise_generator_permution_index_expand = tf.expand_dims(self.pointwise_generator_permution_index,
                                                                         axis=1)  # 形状变为 [bs, 1, 5
        self.pointwise_generator_permution_index_expand = tf.stop_gradient(
            self.pointwise_generator_permution_index_expand)

        self.pointwise_permuation_mask = tf.reshape(
            tf.batch_gather(self.permuation_mask, self.pointwise_generator_permution_index), [-1, POI_NUM, 1])
        self.pointwise_permuation_mask = tf.expand_dims(self.pointwise_permuation_mask, axis=1)  # 形状变为 [bs, 1, 5,1]
        self.pointwise_permuation_mask = tf.stop_gradient(self.pointwise_permuation_mask)
        # self.Q_network_output = self.out

    # 位置分布ListWise
    def _Listwise_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            listwise_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.listwise_cur_page_oam_output, self.listwise_intra_attr_output_each_poi = self._oam(
                listwise_cate_feature_embeddings_stopped)

            self.listwise_behavior_page_oam_output = tf.identity(self.listwise_cur_page_oam_output)
            for i in range(PAGE_NUM):
                listwise_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(
                    self.behavior_cate_feature_embeddings)
                self.listwise_tmp_behavior_input = tf.reshape(
                    tf.gather(listwise_behavior_cate_feature_embeddings_stopped, list(range(i, i + 1)), axis=1),
                    [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_behavior_oam_output, _ = self._oam(self.listwise_tmp_behavior_input)
                if i == 0:
                    self.listwise_behavior_page_oam_output = tf.expand_dims(self.listwise_tmp_behavior_oam_output,
                                                                            axis=1)
                else:
                    self.listwise_behavior_page_oam_output = tf.concat([self.listwise_behavior_page_oam_output,
                                                                        tf.expand_dims(
                                                                            self.listwise_tmp_behavior_oam_output,
                                                                            axis=1)], axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            listwise_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.listwise_user_interest_emb = self._user_interest_target_attention(self.listwise_cur_page_oam_output,
                                                                                   self.listwise_behavior_page_oam_output,
                                                                                   listwise_mask_feature_stopped)

            # features['page_mask'] noindp

            self.listwise_poi_rep_emb = self.listwise_intra_attr_output_each_poi  # -1 * poi * 8
            for i in range(POI_NUM):
                self.listwise_tmp_poi_input = tf.reshape(
                    tf.gather(self.listwise_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.listwise_tmp_poi_output = self._intra_poi_self_attention(self.listwise_tmp_poi_input)
                if i == 0:
                    self.listwise_poi_rep_emb = tf.expand_dims(self.listwise_tmp_poi_output, axis=1)
                else:
                    self.listwise_poi_rep_emb = tf.concat(
                        [self.listwise_poi_rep_emb, tf.expand_dims(self.listwise_tmp_poi_output, axis=1)], axis=1)

            self.listwise_cur_page_oam_output = tf.tile(tf.expand_dims(self.listwise_cur_page_oam_output, axis=1),
                                                        [1, POI_NUM, 1])
            self.listwise_user_interest_emb = tf.tile(tf.expand_dims(self.listwise_user_interest_emb, axis=1),
                                                      [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            listwise_fc_out = tf.reshape(self.cate_feature_embeddings, [-1, POI_NUM, FEATURE_CATE_NUM *
                                                                        CATE_FEATURE_EMBEDDINGS_SHAPE[
                                                                            1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Listwise_MLP_A" + str(i)
                listwise_fc_out = tf.layers.dense(listwise_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i],
                                                  activation=None,
                                                  name=dense_name)
                listwise_fc_out = tf.nn.swish(listwise_fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            listwise_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            listwise_fc_input = tf.concat(
                [self.listwise_cur_page_oam_output, self.listwise_user_interest_emb, self.listwise_poi_rep_emb,
                 listwise_feat_predict_stopped, listwise_fc_out], axis=2)

            listwise_fc_out_ctr, listwise_fc_out_imp = tf.reshape(listwise_fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(
                listwise_fc_input, [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "Listwise_MLP_B" + str(i)
                listwise_fc_out_ctr = tf.layers.dense(listwise_fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i],
                                                      activation=None, name=dense_name)
                listwise_fc_out_ctr = tf.nn.relu(listwise_fc_out_ctr)

            listwise_ctr_out = tf.layers.dense(listwise_fc_out_ctr, OUT_NUM, activation=None,
                                               name="listwise_final_out_ctr")

            if not self.train:
                listwise_ctr_out = tf.nn.sigmoid(listwise_ctr_out)

            self.listwise_generator_out = tf.reshape(listwise_ctr_out, [-1, POI_NUM])  # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.listwise_generator_permution_index = tf.argsort(self.listwise_generator_out, axis=1,
                                                                 direction='DESCENDING')  # 第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.listwise_generator_permution_index = tf.stop_gradient(self.listwise_generator_permution_index)

            self.listwise_generator_embedding = tf.reshape(
                tf.batch_gather(self.cate_feature_embeddings_for_permuation, self.listwise_generator_permution_index),
                [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.listwise_generator_embedding = tf.expand_dims(self.listwise_generator_embedding,
                                                               axis=1)  # [bs, 1, 5,7,8]
            self.listwise_generator_embedding = tf.stop_gradient(self.listwise_generator_embedding)

            self.listwise_generator_permution_index_expand = tf.expand_dims(self.listwise_generator_permution_index,
                                                                            axis=1)  # [bs, 1, 5]
            self.listwise_generator_permution_index_expand = tf.stop_gradient(
                self.listwise_generator_permution_index_expand)

            self.listwise_permuation_mask = tf.reshape(
                tf.batch_gather(self.permuation_mask, self.listwise_generator_permution_index), [-1, POI_NUM, 1])
            self.listwise_permuation_mask = tf.expand_dims(self.listwise_permuation_mask, axis=1)  # [bs,1,5,1]
            self.listwise_permuation_mask = tf.stop_gradient(self.listwise_permuation_mask)
            # self.Q_network_output = self.out

    # 生成式ListWise
    def _headlayers(self, top_tensor_list, position_tensor):
        """
        top_tensor_list: [batch_size, 50 embedding_size]]
        position_tensor: [batch_size , embedding_size]
        output: [batch_size, predict_len, 1]
        """
        position_tensor = tf.tile(tf.expand_dims(position_tensor, axis=1), [1, POI_NUM, 1])
        output = tf.concat([top_tensor_list, position_tensor], axis=2)
        with tf.variable_scope("head_layers", reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(output, 64, activation=tf.nn.relu, name="head_1")
            output = tf.layers.dense(output, 16, activation=tf.nn.relu, name="head_2")
            output = tf.layers.dense(output, 1, activation=None, name="score")
        return output

    def _Generative_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.generative_cur_page_oam_output, self.generative_intra_attr_output_each_poi = self._oam(
                generative_cate_feature_embeddings_stopped)

            self.generative_behavior_page_oam_output = tf.identity(self.generative_cur_page_oam_output)
            for i in range(PAGE_NUM):
                generative_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(
                    self.behavior_cate_feature_embeddings)
                self.generative_tmp_behavior_input = tf.reshape(
                    tf.gather(generative_behavior_cate_feature_embeddings_stopped, list(range(i, i + 1)), axis=1),
                    [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_behavior_oam_output, _ = self._oam(self.generative_tmp_behavior_input)
                if i == 0:
                    self.generative_behavior_page_oam_output = tf.expand_dims(self.generative_tmp_behavior_oam_output,
                                                                              axis=1)
                else:
                    self.generative_behavior_page_oam_output = tf.concat([self.generative_behavior_page_oam_output,
                                                                          tf.expand_dims(
                                                                              self.generative_tmp_behavior_oam_output,
                                                                              axis=1)], axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            generative_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.generative_user_interest_emb = self._user_interest_target_attention(
                self.generative_cur_page_oam_output, self.generative_behavior_page_oam_output,
                generative_mask_feature_stopped)

            self.generative_poi_rep_emb = self.generative_intra_attr_output_each_poi  # -1 * poi * 8
            for i in range(POI_NUM):
                self.generative_tmp_poi_input = tf.reshape(
                    tf.gather(self.generative_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.generative_tmp_poi_output = self._intra_poi_self_attention(self.generative_tmp_poi_input)
                if i == 0:
                    self.generative_poi_rep_emb = tf.expand_dims(self.generative_tmp_poi_output, axis=1)
                else:
                    self.generative_poi_rep_emb = tf.concat(
                        [self.generative_poi_rep_emb, tf.expand_dims(self.generative_tmp_poi_output, axis=1)], axis=1)

            self.generative_cur_page_oam_output = tf.tile(tf.expand_dims(self.generative_cur_page_oam_output, axis=1),
                                                          [1, POI_NUM, 1])
            self.generative_user_interest_emb = tf.tile(tf.expand_dims(self.generative_user_interest_emb, axis=1),
                                                        [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            generative_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            generative_fc_out = tf.reshape(generative_cate_feature_embeddings_stopped, [-1, POI_NUM, FEATURE_CATE_NUM *
                                                                                        CATE_FEATURE_EMBEDDINGS_SHAPE[
                                                                                            1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Generative_MLP_A" + str(i)
                generative_fc_out = tf.layers.dense(generative_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i],
                                                    activation=None,
                                                    name=dense_name)
                generative_fc_out = tf.nn.swish(generative_fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            generative_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            generative_fc_input = tf.concat(
                [self.generative_cur_page_oam_output, self.generative_user_interest_emb, self.generative_poi_rep_emb,
                 generative_feat_predict_stopped, generative_fc_out], axis=2)  # Batch_size * POI_NUM * FEAT_NUM
            generative_fc_input = tf.layers.dense(generative_fc_input, 64, activation=None,
                                                  name="generative_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            batch_size = tf.shape(generative_fc_input)[0]
            with tf.variable_scope("rnn_init_state", reuse=tf.AUTO_REUSE):
                state = tf.reduce_mean(generative_fc_input, axis=1)  # [B,64]
                state = tf.stop_gradient(state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                input_tensor = tf.zeros_like(state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64, name='gru_cell')  # 替换 tf.contrib.rnn.GRUCell

            if self.train:
                self.label = labels['ctr_label']
                self.generative_generator_loss = 0.0
                output_list = []
                label_list = []
                eval_weight_list = []
                generative_generator_out = []
                generative_generator_permution_index = []
                mask_tmp = tf.ones([batch_size, POI_NUM], dtype=tf.float32)  # [B, 5]
                eval_ones = tf.ones([batch_size, POI_NUM], dtype=tf.float32)  # [B, 5]

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor)  # [B, 5,1], head_layer最后一层不用激活函数
                    head_score_origin = tf.reshape(head_score, [-1, POI_NUM])  # [B, 5]
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - mask_tmp, -1.0e9), head_score_origin))  # [B,5]
                    # lable
                    select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1),
                                           [tf.shape(head_score)[0], 1])  # [B, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32)  # [B, 5]

                    # sample_reward = realshow_list * sample_weight
                    label_mask = self.label * select_onehot  # [B, 5], 只有曝光位置有reward, 其他为 0
                    # loss
                    tmp_loss = -1.0 * tf.reduce_sum(label_mask * tf.log(tf.clip_by_value(head_score, 1e-9, 1.0)),
                                                    axis=-1)  # [B]
                    tmp_loss = tf.reduce_mean(tmp_loss)
                    self.generative_generator_loss = self.generative_generator_loss + tmp_loss
                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))

                    output_list.append(head_score)
                    label_list.append(label_mask)
                    weight = mask_tmp * eval_ones
                    eval_weight_list.append(weight)

                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [B, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]
                    input_tensor = tf.stop_gradient(input_tensor)
            else:
                generative_generator_permution_index = []
                generative_generator_out = []
                mask_tmp = tf.ones([batch_size, PAGE_NUM], dtype=tf.float32)  # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    output_tensor, state = gru_cell(input_tensor, state)
                    head_score = self._headlayers(generative_fc_input, output_tensor)  # [B, 50,1]
                    head_score = tf.reshape(head_score, [-1, POI_NUM])
                    head_score = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), head_score))  # [B,50]

                    select_index = tf.reshape(tf.argmax(head_score, axis=1, output_type=tf.int32),
                                              [-1, 1])  # [batch_size, 1]
                    select_onehot = tf.one_hot(tf.reshape(select_index, [-1]), POI_NUM, dtype=tf.float32)
                    mask_tmp = mask_tmp - select_onehot
                    input_tensor = tf.batch_gather(generative_fc_input, select_index)  # [batch_size, 1, embedding_size]
                    input_tensor = tf.reshape(input_tensor, [-1, 64])  # [batch_size, embedding_size]

                    generative_generator_out.append(tf.reduce_max(head_score, axis=1))
                    generative_generator_permution_index.append(tf.argmax(head_score, axis=1, output_type=tf.int32))

            self.generative_generator_out = tf.stack(generative_generator_out, axis=1)  # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.generative_generator_permution_index = tf.stack(generative_generator_permution_index,
                                                                 axis=1)  # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.generative_generator_permution_index = tf.stop_gradient(self.generative_generator_permution_index)

            self.generative_generator_embedding = tf.reshape(
                tf.batch_gather(self.cate_feature_embeddings_for_permuation, self.generative_generator_permution_index),
                [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.generative_generator_embedding = tf.expand_dims(self.generative_generator_embedding,
                                                                 axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.generative_generator_embedding = tf.stop_gradient(self.generative_generator_embedding)

            self.generative_generator_permution_index_expand = tf.expand_dims(self.generative_generator_permution_index,
                                                                              axis=1)  # 形状变为 [bs, 1, 5
            self.generative_generator_permution_index_expand = tf.stop_gradient(
                self.generative_generator_permution_index_expand)

            self.generative_permuation_mask = tf.reshape(
                tf.batch_gather(self.permuation_mask, self.generative_generator_permution_index), [-1, POI_NUM, 1])
            self.generative_permuation_mask = tf.expand_dims(self.generative_permuation_mask,
                                                             axis=1)  # 形状变为 [bs, 1, 5,1]
            self.generative_permuation_mask = tf.stop_gradient(self.generative_permuation_mask)
            # self.Q_network_output = self.out

    # Submodularity Model
    def _Submodularity_generator_model(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            self.submodularity_cur_page_oam_output, self.submodularity_intra_attr_output_each_poi = self._oam(
                submodularity_cate_feature_embeddings_stopped)

            self.submodularity_behavior_page_oam_output = tf.identity(self.submodularity_cur_page_oam_output)
            for i in range(PAGE_NUM):
                submodularity_behavior_cate_feature_embeddings_stopped = tf.stop_gradient(
                    self.behavior_cate_feature_embeddings)
                self.submodularity_tmp_behavior_input = tf.reshape(
                    tf.gather(submodularity_behavior_cate_feature_embeddings_stopped, list(range(i, i + 1)), axis=1),
                    [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.submodularity_tmp_behavior_oam_output, _ = self._oam(self.submodularity_tmp_behavior_input)
                if i == 0:
                    self.submodularity_behavior_page_oam_output = tf.expand_dims(
                        self.submodularity_tmp_behavior_oam_output, axis=1)
                else:
                    self.submodularity_behavior_page_oam_output = tf.concat(
                        [self.submodularity_behavior_page_oam_output,
                         tf.expand_dims(self.submodularity_tmp_behavior_oam_output, axis=1)], axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            submodularity_mask_feature_stopped = tf.stop_gradient(features['page_mask'])
            self.submodularity_user_interest_emb = self._user_interest_target_attention(
                self.submodularity_cur_page_oam_output, self.submodularity_behavior_page_oam_output,
                submodularity_mask_feature_stopped)

            self.submodularity_poi_rep_emb = self.submodularity_intra_attr_output_each_poi  # -1 * poi * 8
            for i in range(POI_NUM):
                self.submodularity_tmp_poi_input = tf.reshape(
                    tf.gather(self.submodularity_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.submodularity_tmp_poi_output = self._intra_poi_self_attention(self.submodularity_tmp_poi_input)
                if i == 0:
                    self.submodularity_poi_rep_emb = tf.expand_dims(self.submodularity_tmp_poi_output, axis=1)
                else:
                    self.submodularity_poi_rep_emb = tf.concat(
                        [self.submodularity_poi_rep_emb, tf.expand_dims(self.submodularity_tmp_poi_output, axis=1)],
                        axis=1)

            self.submodularity_cur_page_oam_output = tf.tile(
                tf.expand_dims(self.submodularity_cur_page_oam_output, axis=1), [1, POI_NUM, 1])
            self.submodularity_user_interest_emb = tf.tile(tf.expand_dims(self.submodularity_user_interest_emb, axis=1),
                                                           [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)
            submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
            submodularity_fc_out = tf.reshape(submodularity_cate_feature_embeddings_stopped, [-1, POI_NUM,
                                                                                              FEATURE_CATE_NUM *
                                                                                              CATE_FEATURE_EMBEDDINGS_SHAPE[
                                                                                                  1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "Submodualrity_MLP_A" + str(i)
                submodularity_fc_out = tf.layers.dense(submodularity_fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i],
                                                       activation=None,
                                                       name=dense_name)
                submodularity_fc_out = tf.nn.swish(submodularity_fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)
            submodularity_feat_predict_stopped = tf.stop_gradient(self.feat_predict)
            submodularity_fc_input = tf.concat(
                [self.submodularity_cur_page_oam_output, self.submodularity_user_interest_emb,
                 self.submodularity_poi_rep_emb, submodularity_feat_predict_stopped, submodularity_fc_out],
                axis=2)  # Batch_size * POI_NUM * FEAT_NUM
            submodularity_fc_input = tf.layers.dense(submodularity_fc_input, 64, activation=None,
                                                     name="submodularity_fc_input_emb")

            # fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input, [-1, MLP_INPUT_DIM])
            submodularity_batch_size = tf.shape(submodularity_fc_input)[0]
            with tf.variable_scope("submodularity_rnn_init_state", reuse=tf.AUTO_REUSE):
                submodularity_state = tf.reduce_mean(submodularity_fc_input, axis=1)  # [B,64]
                submodularity_state = tf.stop_gradient(submodularity_state)
                # state = tf.layers.dense(state, STATE_EMBEDDING_SIZE, activation=None, name="layer_1")
                submodularity_input_tensor = tf.zeros_like(submodularity_state)
                # input_tensor = tf.get_variable('go', (1, STATE_EMBEDDING_SIZE))
                # input_tensor = tf.tile(input_tensor, [batch_size, 1])  # [batch_size, embedding_size]

            with tf.variable_scope("submodularity_rnn", reuse=tf.AUTO_REUSE):
                submodularity_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=64,
                                                                name='submodularity_gru_cell')  # 替换 tf.contrib.rnn.GRUCell

            if self.train:
                self.label = labels['ctr_label']
                self.submodularity_generator_loss = 0.0
                submodularity_output_list = []
                submodularity_label_list = []
                submodularity_eval_weight_list = []
                submodularity_generator_out = []
                submodularity_generator_permution_index = []
                submodularity_mask_tmp = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32)  # [B, 5]
                submodularity_eval_ones = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32)  # [B, 5]

                # # embedding distance
                # submodularity_cate_feature_embeddings_stopped = tf.stop_gradient(self.cate_feature_embeddings)
                # feature_embedding_expanded = tf.expand_dims(submodularity_cate_feature_embeddings_stopped,
                #                                             axis=1)  # [bs, 1, 5, 7, 8]
                # other_list_embedding = tf.stop_gradient(self.generator_permuation_embeddings)  # [bs, k, 5, 7, 8]
                # # 计算 A 和 AA 在 5, 7, 8 维度上的 L2 范数
                # feature_embedding_norm = tf.norm(feature_embedding_expanded, axis=[3, 4],
                #                                  keepdims=True)  # [bs, 1, 5, 1, 1]
                # other_list_embedding_norm = tf.norm(other_list_embedding, axis=[3, 4],
                #                                     keepdims=True)  # [bs, k, 5, 1, 1]
                # feature_embedding_norm = tf.squeeze(feature_embedding_norm, axis=[-1, -2])  # 形状变为 [1024, 3, 5]
                # other_list_embedding_norm = tf.squeeze(other_list_embedding_norm, axis=[-1, -2])  # 形状变为 [1024, 3, 5]
                #
                # # 计算 A 和 AA 的点积
                # similarity = tf.reduce_sum(feature_embedding_expanded * other_list_embedding, axis=[3, 4])  # [bs, k,5]
                # similarity = tf.stop_gradient(similarity)
                # # 计算余弦相似度
                # cosine_similarity = similarity / (
                #     tf.clip_by_value(feature_embedding_norm * other_list_embedding_norm, 1e-4,
                #                      float('inf')))  # [bs, k,5]
                # cosine_similarity = tf.stop_gradient(cosine_similarity)
                # # 对 k 维度取最大值，获取 A 中每个 item 与 AA 中 k 个候选的最大相似度
                # coverage_score = tf.reduce_mean(cosine_similarity, axis=1)  # [bs, 5]
                # coverage_score = tf.stop_gradient(coverage_score)

                for index in range(POI_NUM):
                    discount = 1.0 / np.log2(2.0 + index)
                    submodularity_output_tensor, submodularity_state = submodularity_gru_cell(
                        submodularity_input_tensor, submodularity_state)
                    submodularity_head_score = self._headlayers(submodularity_fc_input,
                                                                submodularity_output_tensor)  # [B, 5,1], head_layer最后一层不用激活函数
                    submodularity_head_score_origin = tf.reshape(submodularity_head_score, [-1, POI_NUM])  # [B, 5]
                    submodularity_head_score = tf.nn.softmax(tf.add(tf.multiply(1.0 - submodularity_mask_tmp, -1.0e9),
                                                                    submodularity_head_score_origin))  # [B,5]
                    # lable
                    submodularity_select_index = tf.tile(tf.expand_dims(tf.constant([index], dtype=tf.int32), axis=1),
                                                         [tf.shape(submodularity_head_score)[0], 1])  # [B, 1]
                    submodularity_select_onehot = tf.one_hot(tf.reshape(submodularity_select_index, [-1]), POI_NUM,
                                                             dtype=tf.float32)  # [B, 5]

                    # sample_reward = realshow_list * sample_weight
                    submodularity_label_mask = self.label * submodularity_select_onehot  # [B, 5], 只有曝光位置有reward, 其他为 0
                    # submodularity_coverage_score = coverage_score * submodularity_select_onehot

                    # loss
                    submodularity_tmp_loss = -1.0 * tf.reduce_sum(
                        # submodularity_label_mask * tf.log(tf.clip_by_value(submodularity_head_score, 1e-6, 1.0)),
                        # axis=-1) - (-1.0 * tf.reduce_sum(
                        # submodularity_coverage_score * tf.log(tf.clip_by_value(submodularity_head_score, 1e-6, 1.0)),
                        # axis=-1))  # [B]
                        submodularity_label_mask * tf.log(tf.clip_by_value(submodularity_head_score, 1e-6, 1.0)),
                        axis=-1)
                    submodularity_tmp_loss = tf.reduce_mean(submodularity_tmp_loss)
                    self.submodularity_generator_loss = self.submodularity_generator_loss + submodularity_tmp_loss
                    submodularity_generator_out.append(tf.reduce_max(submodularity_head_score, axis=1))
                    submodularity_generator_permution_index.append(
                        tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32))

                    submodularity_output_list.append(submodularity_head_score)
                    submodularity_label_list.append(submodularity_label_mask)
                    submodularity_weight = submodularity_mask_tmp * submodularity_eval_ones
                    submodularity_eval_weight_list.append(submodularity_weight)

                    submodularity_mask_tmp = submodularity_mask_tmp - submodularity_select_onehot
                    submodularity_input_tensor = tf.batch_gather(submodularity_fc_input,
                                                                 submodularity_select_index)  # [B, 1, embedding_size]
                    submodularity_input_tensor = tf.reshape(submodularity_input_tensor,
                                                            [-1, 64])  # [batch_size, embedding_size]
                    submodularity_input_tensor = tf.stop_gradient(submodularity_input_tensor)

                submodularity_generator_permution_index_ = tf.stack(submodularity_generator_permution_index,
                                                                         axis=1)  # bs*5
                generator_permuation_index_ = tf.stop_gradient(self.generator_permuation_index)


                submodularity_expanded = tf.expand_dims(submodularity_generator_permution_index_,
                                                        axis=1)  # 形状变为 (?, 1, 5)
                # 拼接
                combined_tensor = tf.concat([submodularity_expanded, generator_permuation_index_],
                                            axis=1)  # 形状变为 (?, 4, 5)
                tensor_expanded = combined_tensor[:, :, :, tf.newaxis]  # 形状变为 (?, 4, 5, 1)
                tensor_expanded_transposed = combined_tensor[:, :, tf.newaxis, :]  # 形状变为 (?, 4, 1, 5)
                # 计算差异矩阵
                difference_matrix = tensor_expanded - tensor_expanded_transposed
                ones = tf.ones_like(difference_matrix, dtype=tf.int32)
                zeros = tf.zeros_like(difference_matrix, dtype=tf.int32)
                result_matrix = tf.where(difference_matrix < 0, ones, zeros)
                final_result = tf.map_fn(
                    lambda matrix: tf.stack([
                        tf.abs(matrix[i] - matrix[j])
                        for i in range(4) for j in range(i + 1, 4)
                    ]),
                    result_matrix,
                    dtype=tf.int32
                )
                # 统计每个 (5, 5) 差值矩阵中值为1的元素的数量
                count_of_ones = tf.reduce_sum(tf.cast(tf.equal(final_result, 1), tf.int32), axis=[2, 3])
                LC_score = tf.reduce_sum(count_of_ones, axis=-1)
                coverage_score = tf.reduce_mean(LC_score)
                coverage_score = tf.cast(coverage_score, tf.float32)
                submodularity_coverage_loss = coverage_score
                # submodularity_coverage_loss = tf.Print(submodularity_coverage_loss, [submodularity_coverage_loss],
                                                       # "Submodularity Coverage Loss: ")
                self.submodularity_generator_loss = self.submodularity_generator_loss - 1*submodularity_coverage_loss
                # self.submodularity_generator_loss = tf.Print(self.submodularity_generator_loss,
                #                                              [self.submodularity_generator_loss],
                #                                              "Submodularity Generator Loss: ")


            else:
                submodularity_generator_permution_index = []
                submodularity_generator_out = []
                submodularity_mask_tmp = tf.ones([submodularity_batch_size, POI_NUM], dtype=tf.float32)  # [B, 5]
                ## 这边预测6个
                for index in range(POI_NUM):
                    submodularity_output_tensor, submodularity_state = submodularity_gru_cell(
                        submodularity_input_tensor, submodularity_state)
                    submodularity_head_score = self._headlayers(submodularity_fc_input,
                                                                submodularity_output_tensor)  # [B, 5,1]
                    submodularity_head_score = tf.reshape(submodularity_head_score, [-1, POI_NUM])
                    submodularity_head_score = tf.nn.softmax(
                        tf.add(tf.multiply(1. - submodularity_mask_tmp, -1.0e9), submodularity_head_score))  # [B,5]

                    submodularity_select_index = tf.reshape(
                        tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32), [-1, 1])  # [batch_size, 1]
                    submodularity_select_onehot = tf.one_hot(tf.reshape(submodularity_select_index, [-1]), POI_NUM,
                                                             dtype=tf.float32)
                    submodularity_mask_tmp = submodularity_mask_tmp - submodularity_select_onehot
                    submodularity_input_tensor = tf.batch_gather(submodularity_fc_input,
                                                                 submodularity_select_index)  # [batch_size, 1, embedding_size]
                    submodularity_input_tensor = tf.reshape(submodularity_input_tensor,
                                                            [-1, 64])  # [batch_size, embedding_size]

                    submodularity_generator_out.append(tf.reduce_max(submodularity_head_score, axis=1))
                    submodularity_generator_permution_index.append(
                        tf.argmax(submodularity_head_score, axis=1, output_type=tf.int32))

            self.submodularity_generator_out = tf.stack(submodularity_generator_out, axis=1)  # bs*5
            # self.listwise_generator_max_pred_value = tf.reduce_max(self.listwise_generator_out, axis=1) # bs*1
            # self.listwise_generator_index = tf.argmax(self.listwise_generator_out, axis=1) #bs*1
            # 使用 tf.argsort 来获取排序后的索引
            self.submodularity_generator_permution_index = tf.stack(submodularity_generator_permution_index,
                                                                    axis=1)  # bs*5 #第二维度为排序后对应的index，比如第二维度的分数为[0.1,0.2,0,3,0.4,0.5]那么第二维度对应的index为[4,3,2,1,0]
            self.submodularity_generator_permution_index = tf.stop_gradient(
                self.submodularity_generator_permution_index)

            self.submodularity_generator_embedding = tf.reshape(
                tf.batch_gather(self.cate_feature_embeddings_for_permuation,
                                self.submodularity_generator_permution_index),
                [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
            self.submodularity_generator_embedding = tf.expand_dims(self.submodularity_generator_embedding,
                                                                    axis=1)  # 形状变为 [bs, 1, 5,7,8]
            self.submodularity_generator_embedding = tf.stop_gradient(self.submodularity_generator_embedding)

            self.submodularity_generator_permution_index_expand = tf.expand_dims(
                self.submodularity_generator_permution_index, axis=1)  # 形状变为 [bs, 1, 5
            self.submodularity_generator_permution_index_expand = tf.stop_gradient(
                self.submodularity_generator_permution_index_expand)

            self.submodularity_permuation_mask = tf.reshape(
                tf.batch_gather(self.permuation_mask, self.submodularity_generator_permution_index), [-1, POI_NUM, 1])
            self.submodularity_permuation_mask = tf.expand_dims(self.submodularity_permuation_mask,
                                                                axis=1)  # 形状变为 [bs, 1, 5,1]
            self.submodularity_permuation_mask = tf.stop_gradient(self.submodularity_permuation_mask)

    def _OCPM(self, features, labels, mode, params):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('dnn_model'):
            self.cur_page_oam_output, self.intra_attr_output_each_poi = self._oam(self.cate_feature_embeddings)

            self.behavior_page_oam_output = tf.identity(self.cur_page_oam_output)
            for i in range(PAGE_NUM):
                self.tmp_behavior_input = tf.reshape(
                    tf.gather(self.behavior_cate_feature_embeddings, list(range(i, i + 1)), axis=1),
                    [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_behavior_oam_output, _ = self._oam(self.tmp_behavior_input)
                if i == 0:
                    self.behavior_page_oam_output = tf.expand_dims(self.tmp_behavior_oam_output, axis=1)
                else:
                    self.behavior_page_oam_output = tf.concat(
                        [self.behavior_page_oam_output, tf.expand_dims(self.tmp_behavior_oam_output, axis=1)], axis=1)

            # self.behavior_page_oam_output = tf.Print(self.behavior_page_oam_output, [self.behavior_page_oam_output],
            #                                          'self.behavior_page_oam_output:', summarize=100)
            # tf.identity(self.behavior_page_oam_output)
            self.user_interest_emb = self._user_interest_target_attention(self.cur_page_oam_output,
                                                                          self.behavior_page_oam_output,
                                                                          features['page_mask'])

            self.poi_rep_emb = self.intra_attr_output_each_poi  # -1 * poi * 8
            for i in range(POI_NUM):
                self.tmp_poi_input = tf.reshape(
                    tf.gather(self.intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                self.tmp_poi_output = self._intra_poi_self_attention(self.tmp_poi_input)
                if i == 0:
                    self.poi_rep_emb = tf.expand_dims(self.tmp_poi_output, axis=1)
                else:
                    self.poi_rep_emb = tf.concat([self.poi_rep_emb, tf.expand_dims(self.tmp_poi_output, axis=1)],
                                                 axis=1)

            self.cur_page_oam_output = tf.tile(tf.expand_dims(self.cur_page_oam_output, axis=1), [1, POI_NUM, 1])
            self.user_interest_emb = tf.tile(tf.expand_dims(self.user_interest_emb, axis=1), [1, POI_NUM, 1])

            # self.cur_page_oam_output = tf.Print(self.cur_page_oam_output, [self.cur_page_oam_output],
            #                                     'self.cur_page_oam_output:', summarize=100)
            # tf.identity(self.cur_page_oam_output)
            #
            #
            # self.user_interest_emb = tf.Print(self.user_interest_emb, [self.user_interest_emb],
            #                                     'self.user_interest_emb:', summarize=100)
            # tf.identity(self.user_interest_emb)
            # self.poi_rep_emb = tf.Print(self.poi_rep_emb, [self.poi_rep_emb],
            #                                     'self.poi_rep_emb:', summarize=100)
            # tf.identity(self.poi_rep_emb)

            fc_out = tf.reshape(self.cate_feature_embeddings, [-1, POI_NUM,
                                                               FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[
                                                                   1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name)
                fc_out = tf.nn.swish(fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat(
                [self.cur_page_oam_output, self.user_interest_emb, self.poi_rep_emb, self.feat_predict, fc_out], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input,
                                                                                           [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i], activation=None,
                                             name=dense_name)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_model_final_out_ctr")

            if not self.train:
                ctr_out = tf.nn.sigmoid(ctr_out)

            self.out = tf.reshape(ctr_out, [-1, POI_NUM])
            self.Q_network_output = self.out

    def _OCPM_TOP_K(self, features, labels, mode, params, k_page, k_predict, k, mask):
        self.train = (mode == tf.estimator.ModeKeys.TRAIN)

        target_page_embedding = tf.reshape(k_page, [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
        target_page_predict_fea = tf.reshape(k_predict, [-1, POI_NUM, 1])

        with tf.name_scope('dnn_model'):
            cur_page_oam_output, target_intra_attr_output_each_poi = self._oam(target_page_embedding)

            behavior_page_oam_output = self.behavior_page_oam_output

            user_interest_emb = self._user_interest_target_attention_topK(cur_page_oam_output, behavior_page_oam_output,
                                                                          features['page_mask'], k)

            poi_rep_emb = target_intra_attr_output_each_poi  # （batch * k） * poi * 8
            for i in range(POI_NUM):
                tmp_poi_input = tf.reshape(
                    tf.gather(target_intra_attr_output_each_poi, list(range(i, i + 1)), axis=1),
                    [-1, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])
                tmp_poi_output = self._intra_poi_self_attention(tmp_poi_input)
                if i == 0:
                    poi_rep_emb = tf.expand_dims(tmp_poi_output, axis=1)
                else:
                    poi_rep_emb = tf.concat(
                        [poi_rep_emb, tf.expand_dims(tmp_poi_output, axis=1)], axis=1)

            cur_page_oam_output = tf.tile(tf.expand_dims(cur_page_oam_output, axis=1), [1, POI_NUM, 1])
            user_interest_emb = tf.tile(tf.expand_dims(user_interest_emb, axis=1), [1, POI_NUM, 1])

            fc_out = tf.reshape(target_page_embedding, [-1, POI_NUM, FEATURE_CATE_NUM * CATE_FEATURE_EMBEDDINGS_SHAPE[
                1]])  # Batch_size * POI_NUM * FEAT_NUM
            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'])):
                dense_name = "MLP_A" + str(i)
                fc_out = tf.layers.dense(fc_out, MODEL_PARAMS['INPUT_TENSOR_LAYERS_A'][i], activation=None,
                                         name=dense_name, reuse=tf.AUTO_REUSE)
                fc_out = tf.nn.swish(fc_out)

            # fc_out = tf.concat([fc_out, self.feat_predict], axis=2)

            fc_input = tf.concat(
                [cur_page_oam_output, user_interest_emb, poi_rep_emb, target_page_predict_fea, fc_out], axis=2)

            fc_out_ctr, fc_out_imp = tf.reshape(fc_input, [-1, MLP_INPUT_DIM]), tf.reshape(fc_input,
                                                                                           [-1, MLP_INPUT_DIM])

            for i in range(0, len(MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'])):
                dense_name = "MLP_B" + str(i)
                fc_out_ctr = tf.layers.dense(fc_out_ctr, MODEL_PARAMS['INPUT_TENSOR_LAYERS_B'][i],
                                             activation=None, name=dense_name, reuse=tf.AUTO_REUSE)
                fc_out_ctr = tf.nn.relu(fc_out_ctr)

            ctr_out = tf.layers.dense(fc_out_ctr, OUT_NUM, activation=None, name="evaluator_topk_final_out_ctr",
                                      reuse=tf.AUTO_REUSE)
            ctr_out = tf.nn.sigmoid(ctr_out)

            topk_out = tf.reshape(ctr_out, [-1, k, POI_NUM])
            mask = tf.reshape(mask, [-1, k, POI_NUM])
            return topk_out * mask

    def _create_loss(self, labels):
        with tf.name_scope('loss'):
            self.label = labels['ctr_label']
            self.mask = labels['mask']
            # ctr_loss
            self.pointwise_generator_loss = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(self.label, self.pointwise_generator_out, weights=self.mask))
            self.listwise_generator_loss = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(self.label, self.listwise_generator_out, weights=self.mask))
            # self.submodularity_generator_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.submodularity_generator_out, weights=self.mask))
            self.generator_loss = self.pointwise_generator_loss + self.listwise_generator_loss + self.generative_generator_loss + self.submodularity_generator_loss
            self.evaluator_loss = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(self.label, self.out, weights=self.mask))
            self.loss = self.generator_loss + self.evaluator_loss
            # if USE_CONSTRATIVE_LOSS:
            #     self.loss -= CONSTRATIVE_LOSS_K * (tf.reduce_mean(self.topK_output,axis=[0,1,2]) - tf.reduce_mean(self.lastK_output,axis=[0,1,2]))

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def _create_weights(self):
        with tf.name_scope('feature_emb_weights'):
            self.feature_weights = {
                'embedding': tf.get_variable('embedding',
                                             shape=CATE_FEATURE_EMBEDDINGS_SHAPE,
                                             initializer=tf.zeros_initializer(),
                                             )
            }

            glorot = np.sqrt(2.0 / (4 + 1))

            self.intra_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * POI_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.inter_attr_self_attention_weights = {
                'weight_Q': tf.get_variable(name='inter_attr_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='inter_attr_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='inter_attr_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='inter_attr_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

            self.intra_poi_self_attention_weights = {
                'weight_Q': tf.get_variable(name='intra_poi_self_attention_weights_Q',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_K': tf.get_variable(name='intra_poi_self_attention_weights_K',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_V': tf.get_variable(name='intra_poi_self_attention_weights_V',
                                            shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1], CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                            initializer=tf.random_normal_initializer(0.0, glorot),
                                            dtype=np.float32),
                'weight_MLP': tf.get_variable(name='intra_poi_self_attention_weights_MLP',
                                              shape=[CATE_FEATURE_EMBEDDINGS_SHAPE[1] * FEATURE_CATE_NUM,
                                                     CATE_FEATURE_EMBEDDINGS_SHAPE[1]],
                                              initializer=tf.random_normal_initializer(0.0, glorot),
                                              dtype=np.float32)
            }

    def _process_features(self, features):
        # env_feature = > dense_feature
        # cxr_feature = > screen_predict_feature
        # cat_feature = > screen_cate_feature
        # dense_feature = > screen_dense_feature

        # N * M * K
        # N * D ( D <= M )
        cate_fea = tf.reshape(features['cate_feature'], [-1, POI_NUM * FEATURE_CATE_NUM])
        self.cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], cate_fea),
            [-1, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        full_permuation_index = tf.reshape(features['full_permuation_index'], [PERMUATION_SIZE, POI_NUM])

        self.cate_feature_embeddings_for_permuation = tf.reshape(self.cate_feature_embeddings, [-1, POI_NUM,
                                                                                                FEATURE_CATE_NUM *
                                                                                                CATE_FEATURE_EMBEDDINGS_SHAPE[
                                                                                                    1]])
        self.full_permuation_index = tf.reshape(tf.tile(tf.expand_dims(full_permuation_index, axis=0),
                                                        [tf.shape(self.cate_feature_embeddings_for_permuation)[0], 1,
                                                         1]), [-1, POI_NUM * PERMUATION_SIZE])
        self.full_permuation_embeddings = tf.reshape(
            tf.batch_gather(self.cate_feature_embeddings_for_permuation, self.full_permuation_index),
            [-1, PERMUATION_SIZE, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        # self.input_embedding = tf.concat(
        #     [self.cate_feature_embeddings], axis=2)
        self.feat_predict = features['dense_feature']
        self.feat_predict_for_permuation = tf.reshape(self.feat_predict, [-1, POI_NUM, 1])

        self.full_permuation_feat_predict = tf.reshape(
            tf.batch_gather(self.feat_predict_for_permuation, self.full_permuation_index),
            [-1, PERMUATION_SIZE, POI_NUM, 1])

        self.permuation_mask = tf.reshape(features['permuation_mask'], [-1, POI_NUM, 1])
        self.full_permuation_mask = tf.reshape(tf.batch_gather(self.permuation_mask, self.full_permuation_index),
                                               [-1, PERMUATION_SIZE, POI_NUM, 1])

        behavior_cate_fea = tf.reshape(features['behavior_cate_feature'], [-1, PAGE_NUM * POI_NUM * FEATURE_CATE_NUM])
        self.behavior_cate_feature_embeddings = tf.reshape(tf.nn.embedding_lookup(
            self.feature_weights['embedding'], behavior_cate_fea),
            [-1, PAGE_NUM, POI_NUM, FEATURE_CATE_NUM, CATE_FEATURE_EMBEDDINGS_SHAPE[1]])

        self.behavior_input_embedding = tf.concat([self.behavior_cate_feature_embeddings], axis=3)
        self.behavior_feat_predict = features['behavior_dense_feature']

        return features

    def _create_indicator(self, labels):
        ctr_out = self.out
        ctr_out = tf.reduce_mean(tf.nn.sigmoid(ctr_out))

        # All gradients of loss function wrt trainable variables
        '''
        grads = tf.gradients(self.loss, tf.trainable_variables())
        for grad, var in list(zip(grads, tf.trainable_variables())):
            tf.summary.histogram(var.name + '/gradient', grad)
        '''

        def format_log(tensors):
            log0 = "train info: step {}, loss={:.4f}, ctr_loss={:.4f}, " \
                   "ctr_out={:.4f},  avg_ctr_topk={:.4f},  avt_ctr_lastk={:.4f}".format(
                tensors["step"], tensors["loss"], tensors["loss"],
                tensors["ctr_out"], tensors['avg_ctr_topk'], tensors['avt_ctr_lastk']
            )
            return log0

        self.logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(),
                                                        "loss": self.loss,
                                                        "ctr_out": ctr_out,
                                                        "avg_ctr_topk": tf.reduce_mean(self.topK_output,
                                                                                       axis=[0, 1, 2]),
                                                        "avt_ctr_lastk": tf.reduce_mean(self.lastK_output,
                                                                                        axis=[0, 1, 2]),
                                                        },
                                                       every_n_iter=5,
                                                       formatter=format_log)

    def model_fn_estimator(self, features, labels, mode, params):
        self._create_weights()
        self._process_features(features)
        self._FGSM(features)
        # pointwise输出一个序列
        self._Pointwise_generator_model(features, labels, mode, params)
        # 位置分布输出一个序列
        self._Listwise_generator_model(features, labels, mode, params)
        # GRN输出一个序列
        self._Generative_generator_model(features, labels, mode, params)

        self.generator_permuation_embeddings = tf.concat(
            [self.pointwise_generator_embedding, self.listwise_generator_embedding,
             self.generative_generator_embedding], axis=1)  # [bs, k, 5,7,8]
        self.generator_permuation_embeddings = tf.stop_gradient(self.generator_permuation_embeddings)

        self.pointwise_generator_out_expand = tf.expand_dims(self.pointwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.listwise_generator_out_expand = tf.expand_dims(self.listwise_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generative_generator_out_expand = tf.expand_dims(self.generative_generator_out, axis=1)  # 形状变为 [bs, 1, 5]
        self.generator_permuation_predict_fea = tf.concat(
            [self.pointwise_generator_out_expand, self.listwise_generator_out_expand,
             self.generative_generator_out_expand], axis=1)  # [bs, k, 5]
        self.generator_permuation_predict_fea = tf.stop_gradient(self.generator_permuation_predict_fea)

        self.generator_permuation_index = tf.concat(
            [self.pointwise_generator_permution_index_expand, self.listwise_generator_permution_index_expand,
             self.generative_generator_permution_index_expand], axis=1)
        self.generator_permuation_index = tf.stop_gradient(self.generator_permuation_index)

        self.generator_permuation_mask = tf.concat(
            [self.pointwise_permuation_mask, self.listwise_permuation_mask, self.generative_permuation_mask],
            axis=1)  # [bs, k, 5,1]
        self.generator_permuation_mask = tf.stop_gradient(self.generator_permuation_mask)

        # 全面性模型
        self._Submodularity_generator_model(features, labels, mode, params)

        self.submodularity_generator_out_expand = tf.expand_dims(self.submodularity_generator_out, axis=1)  # [bs, 1, 5]
        self.submodularity_generator_permuation_embeddings = tf.concat(
            [self.generator_permuation_embeddings, self.submodularity_generator_embedding], axis=1)  # [bs, k, 5,7,8]
        self.submodularity_generator_permuation_embeddings = tf.stop_gradient(
            self.submodularity_generator_permuation_embeddings)

        self.submodularity_generator_permuation_predict_fea = tf.concat(
            [self.generator_permuation_predict_fea, self.submodularity_generator_out_expand], axis=1)  # [bs, k, 5]
        self.submodularity_generator_permuation_predict_fea = tf.stop_gradient(
            self.submodularity_generator_permuation_predict_fea)

        self.submodularity_generator_permuation_index = tf.concat(
            [self.generator_permuation_index, self.submodularity_generator_permution_index_expand],
            axis=1)  # [bs, k, 5]
        self.submodularity_generator_permuation_index = tf.stop_gradient(self.submodularity_generator_permuation_index)

        self.submodularity_generator_permuation_mask = tf.concat(
            [self.generator_permuation_mask, self.submodularity_permuation_mask], axis=1)  # [bs, k, 5,1]
        self.submodularity_generator_permuation_mask = tf.stop_gradient(self.submodularity_generator_permuation_mask)

        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.pointwise_generator_embedding, self.pointwise_generator_out_expand,1,self.pointwise_permuation_mask)
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.listwise_generator_embedding, self.listwise_generator_out_expand,1,self.listwise_permuation_mask)
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generative_generator_out_expand,1,self.generative_permuation_mask)
        self._OCPM(features, labels, mode, params)
        generator_num = 4
        # self.topK_output = self._OCPM_TOP_K(features, labels, mode, params, self.generator_permuation_embeddings, self.generator_permuation_predict_fea,generator_num,self.generator_permuation_mask)
        self.topK_output = self._OCPM_TOP_K(features, labels, mode, params,
                                            self.submodularity_generator_permuation_embeddings,
                                            self.submodularity_generator_permuation_predict_fea, generator_num,
                                            self.submodularity_generator_permuation_mask)
        self.lastK_output = self._OCPM_TOP_K(features, labels, mode, params, self.last_k_permuation_embeddings,
                                             self.last_k_permuation_predict_fea, TOP_K, self.last_k_permuation_mask)
        self.randK_output = self._OCPM_TOP_K(features, labels, mode, params, self.random_k_permuation_embeddings,
                                             self.random_k_permuation_predict_fea, TOP_K, self.random_k_permuation_mask)
        self.beamK_output = self._OCPM_TOP_K(features, labels, mode, params, self.beam_search_k_permuation_embeddings,
                                             self.beam_search_k_permuation_predict_fea, TOP_K,
                                             self.beam_search_k_permuation_mask)
        self.allK_output = self._OCPM_TOP_K(features, labels, mode, params, self.full_permuation_embeddings,
                                            self.full_permuation_feat_predict, PERMUATION_SIZE,
                                            self.full_permuation_mask)

        # 输出avg-ctr最高的序列
        self.avg_ctr_of_topK = tf.reduce_mean(self.topK_output, axis=2)
        max_ctr_index_in_topk = tf.nn.top_k(self.avg_ctr_of_topK, k=1).indices  # batch * 1
        self.final_rerank_output_index = tf.batch_gather(self.submodularity_generator_permuation_index,
                                                         max_ctr_index_in_topk)  # bs,1,5
        # self.max_ctr_index = tf.batch_gather(self.top_k_permutation_indicces,self.max_ctr_index_in_topk) # batch * 1
        # full_permuation_index = tf.reshape(features['full_permuation_index'], [PERMUATION_SIZE, POI_NUM])
        # full_permuation_index = tf.tile(tf.expand_dims(full_permuation_index, axis=0),[tf.shape(self.cate_feature_embeddings_for_permuation)[0], 1, 1]) # batch * 120 * 5

        if self.train:
            self._create_loss(labels)
            self._create_optimizer()
            self._create_indicator(labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op,
                                              training_hooks=[self.logging_hook])
        else:
            if 'save_model' in list(params.keys()):
                outputs = {
                    "Q_network_output": tf.identity(self.Q_network_output, "Q_network_output"),
                    "out": tf.identity(self.out, "out"),
                    'output_index': tf.identity(self.final_rerank_output_index, "out_index")
                }
            else:
                ctr_out = self.out
                # gmv
                outputs = {'out': self.out,
                           'mask': features['mask'],
                           'ctr_out': ctr_out,
                           'ctr_label': features['ctr_label'],
                           'q_out': self.Q_network_output,
                           'cxr_feature': features['dense_feature'],
                           "ctr_topk": self.topK_output,
                           "ctr_lastk": self.lastK_output,
                           "ctr_randk": self.randK_output,
                           "ctr_all": self.allK_output,
                           "ctr_beamk": self.beamK_output,
                           'out_index': self.final_rerank_output_index
                           }
            export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                  tf.estimator.export.PredictOutput(outputs)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs, export_outputs=export_outputs)

    def tf_print(self, var, varStr='null'):
        return tf.Print(var, [var], message=varStr, summarize=100)
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/run.sh
````bash
#!/usr/bin/env bash

LOG_PATH="./"
if [[ ! -d ${LOG_PATH} ]]; then
        mkdir ${LOG_PATH}
fi

project_path=$(cd `dirname $0`; pwd)
project_name="${project_path##*/}"
begin_time=$(date "+%Y-%m-%d %H:%M:%S")
author="yangfan129"

LOG_FILENAME="log"
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --author ${author} --project ${project_name} --begintime ${begin_time} > ${LOG_PATH}${LOG_FILENAME} 2>&1 &
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/tools.py
````python
# -*- coding: utf-8 -*-
import datetime
import time
import os

import math
import random

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(("*" * 50 + " {} START!!!! ".format(self.process_name) + "*" * 50))
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            duration_seconds = end_time - self.begin_time
            duration = str(datetime.timedelta(seconds=duration_seconds))

            print(("#" * 50 + " {} END... time lapsing {}  ".format(self.process_name, duration) + "#" * 50))


class FeatureInfo:
    def __init__(self, feature_info_str):
        self.feature_info_str = feature_info_str

        self.feature_name = "NonFeaName"
        self.feature_size = 0
        self.feature_mask = 1
        self.parse_info_flag = False
        self.part_num = 3

        self._parse_info()

    def _parse_info(self):
        infoList = self.feature_info_str.split()

        if len(infoList) == self.part_num:
            self.feature_name = infoList[0]
            self.feature_size = int(infoList[1])
            self.feature_mask = int(infoList[2])
            self.parse_info_flag = True


def parse_mask_file(feature_mask_file):
    try:
        if not os.path.exists(feature_mask_file):
            print("parse_mask_file fail - file not exists:", feature_mask_file)
            return [], False
        # feature_name_list = []
        feature_mask_list = []
        feature_hold_cnt = 0

        with open(feature_mask_file) as f:
            str_list = f.readlines()

        for i in range(0, len(str_list)):
            str_list[i] = str_list[i].strip('\n').strip()
            if str_list[i] == "":
                continue

            info = FeatureInfo(str_list[i])
            if not info.parse_info_flag:
                print("parse_mask_file fail - parse_info fail:", str_list[i])
                parse_mask_flag = False
                return [], parse_mask_flag

            for j in range(info.feature_size):
                feature_mask_list.append(info.feature_mask)
                if info.feature_mask != 0:
                    feature_hold_cnt += 1
                # if info.feature_size > 1:
                #     feature_name_list.append(info.feature_name + "_" + str(j))
                # else:
                #     feature_name_list.append(info.feature_name)

        parse_mask_flag = True
        return feature_mask_list, parse_mask_flag, feature_hold_cnt
    except Exception as e:
        print("parse_mask_file fail - Exception:", e)
        return [], False

import itertools


# 利用itertools库中的permutations函数,给定一个排列,输出他的全排列
def allPermutation(n):
    permutation = []
    # 首先需要初始化一个1-n的排列
    for i in range(n):
        permutation.append(i)
    # itertools.permutations返回的只是一个对象,需要将其转化成list
    # 每一种排列情况以元组类型存储
    all_permutation = list(itertools.permutations(permutation))
    return all_permutation


def random_vector():
    print([[random.random() for x in list(range(0,8))] for y in list(range(0,5))])




if __name__ == "__main__":
    # feature_mask_list, parse_feature_mask_flag, feature_hold_cnt = parse_mask_file("feature_mask")
    # print(feature_mask_list)
    # print(len(feature_mask_list))
    # print(parse_feature_mask_flag)
    # print(feature_hold_cnt)
    # print(allPermutation(5))
    random_vector()
````

## File: rerank_model_dir/MultiG_E_Submodularity_Model_lc/util.py
````python
import tensorflow as tf

def index_matrix_to_pairs(index_matrix):
    # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
    #                        [[0, 2], [1, 3], [2, 1]]]
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
    rank = len(index_matrix.get_shape())
    if rank == 2:
        replicated_first_indices = tf.tile(
            tf.expand_dims(replicated_first_indices, axis=1),
            [1, tf.shape(index_matrix)[1]])
        replicated_first_indices = tf.cast(replicated_first_indices, dtype=tf.int64)
    return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def string_hash_to_index(tensor, bucket=1<<22):
    return tf.strings.to_hash_bucket_fast(tensor, bucket)

def int_to_string_with_key(tensor, key):
    return key + "_" + tf.strings.as_string(tensor)

def float_to_string_with_key(tensor, key, precision=1):
    return key + "_" + tf.strings.as_string(tensor, precision)

def float_to_int(tensor, order):
    wc = 10 ** order   
    return tf.cast(tensor * wc, tf.int64)

def float_custom_hash(tensor, key, precision=0, bucket=1<<22):
    tensor = float_to_string_with_key(tensor, key, precision)
    tensor = string_hash_to_index(tensor, bucket)
    return tensor

def int_custom_hash(tensor, key, bucket=1<<22):
    tensor = int_to_string_with_key(tensor, key)
    tensor = string_hash_to_index(tensor, bucket)
    return tensor
````

## File: CORE_CODE.md
````markdown
# LC_MGE 核心代码导航

## 打包范围

主实现位于 `rerank_model_dir/MultiG_E_Submodularity_Model_lc/`。
核心包保留其训练、预测评估、数据解析和模型依赖，以及两个直接相关版本的模型与配置，方便理解实现差异。
本次整理通过文件筛选和导航完成，原有源码内容与目录结构保持不变。

| 目录 | 核心包中的内容 | 用途 |
| --- | --- | --- |
| `MultiG_E_Submodularity_Model_lc/` | `main.py`、`model.py`、`config.py`、`data_input.py`、`layers.py`、`tools.py`、`util.py`、`run.sh` | LC 版本主流程及其本地 Python 模块依赖 |
| `MultiG_E_Submodularity_Model/` | `model.py`、`config.py` | 含 Submodularity 生成器的相关版本，用于代码对照 |
| `MultiG_E_model/` | `model.py`、`config.py` | 多生成器与评估器的相关版本，用于代码对照 |

表中的目录均位于 `rerank_model_dir/` 下。两个对照版本仅打包模型与配置；需要运行它们时，应使用原目录中的完整代码。

## 建议阅读顺序

以下文件名均指 LC 主实现目录中的文件。

1. `config.py`：数据路径、训练模式、特征维度、候选排列数量和网络参数。
2. `data_input.py`：TFRecord 解析、特征与标签构造、排列生成及输入流水线。
3. `main.py`：Estimator 创建、训练、预测评估和导出入口；`calculate_result` 汇总预测结果。
4. `model.py`：模型主体。先看 `model_fn_estimator`，再阅读各个生成器与评估器。
5. `layers.py`、`util.py`、`tools.py`：网络层、张量工具、排列枚举和计时工具。

`model.py` 中的主要入口：

| 方法 | 作用 |
| --- | --- |
| `model_fn_estimator` | 串联特征处理、生成器、候选序列评估和训练/预测输出 |
| `_process_features`、`_create_weights` | 特征处理与权重初始化 |
| `_oam`、各类 attention 方法 | 页内属性与用户行为表示 |
| `_Pointwise_generator_model` | Pointwise 生成器 |
| `_Listwise_generator_model` | Listwise 生成器 |
| `_Generative_generator_model` | 自回归序列生成器 |
| `_Submodularity_generator_model` | Submodularity 生成器及 LC 相关计算 |
| `_OCPM`、`_OCPM_TOP_K` | 原始页面与候选序列的评估 |
| `_create_loss`、`_create_optimizer` | 损失组合与优化 |

## 未打包的内容

- DNN、DCN、PRM、EXTR、KuaiShou、PIER、PRS、NAR、seq2slate 等其他基线目录。
- 名称含 `metric` 的独立实验变体。
- 独立的 `evaluate.py`、`model_load.py`、空的 `restore.py` 和依赖外部模板的 `build_model.sh`；主流程的预测评估与导出代码仍保留在 `main.py` 中。
- `avg_std/` 辅助数据、缓存、字节码、日志、模型产物和 ZIP 归档。
- 旧的全量 Repomix 产物 `rerank_model_dir/rm.md`，避免重复嵌套打包。

## 重新生成单文件

在项目根目录执行：

```bash
repomix --config repomix.config.json
```

输出为根目录的 `repomix-output.md`，包含目录树、文件路径和以代码块展示的完整源文件内容。
包不压缩代码，保留注释和内部空行，文件首尾空白由 Repomix 规范化。
调整 `repomix.config.json` 中的 `include` 列表即可修改范围。

## 原代码运行入口

原 README 标注环境为 Python 3 + TensorFlow 1.5；源码使用 TensorFlow 1.x 风格接口，并依赖 NumPy、scikit-learn。本次打包不验证训练环境兼容性。
运行前需在 LC 目录的 `config.py` 中设置本机数据路径和 `TRAIN_MODE`：1 为训练，2 为预测评估，3 为训练后预测评估，4 为导出。

```bash
cd rerank_model_dir/MultiG_E_Submodularity_Model_lc
python main.py
```

也可使用原有 `bash run.sh` 后台运行，输出写入该目录的 `log`。
合并后的 Markdown 是代码阅读与分析材料；执行训练仍使用原始 Python 文件。
````

## File: README.md
````markdown
# LC_MGE_code
**PIER code** contains DNN, DCN, PRM, EXTR, KuaiShou re-rankding, PIER, PRS, LC_MGE

Run Env: python3 + tf1.5

Run code: sh run.sh

Dataset: https://drive.google.com/drive/folders/1BRkP9YPiU1bdviLjo3jrYdXNTqrJAvJM?usp=share_link

核心代码导航与打包范围见 [CORE_CODE.md](CORE_CODE.md)。在项目根目录执行 `repomix --config repomix.config.json`，生成单文件 `repomix-output.md`。
````

## File: repomix.config.json
````json
{
  "output": {
    "filePath": "repomix-output.md",
    "style": "markdown",
    "parsableStyle": true,
    "headerText": "LC_MGE core code. Start with CORE_CODE.md. The LC implementation includes its training/input/model dependencies; the two related versions include model.py and config.py for comparison only. Code bodies, comments, and interior blank lines are preserved; Repomix normalizes boundary whitespace.",
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "git": {
      "sortByChanges": false
    }
  },
  "include": [
    "README.md",
    "CORE_CODE.md",
    "repomix.config.json",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/config.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/data_input.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/layers.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/main.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/model.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/tools.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/util.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model_lc/run.sh",
    "rerank_model_dir/MultiG_E_model/config.py",
    "rerank_model_dir/MultiG_E_model/model.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model/config.py",
    "rerank_model_dir/MultiG_E_Submodularity_Model/model.py"
  ],
  "ignore": {
    "useGitignore": false,
    "useDotIgnore": false,
    "useDefaultPatterns": true,
    "customPatterns": [
      "**/__pycache__/**",
      "**/*.pyc",
      "**/.DS_Store",
      "**/avg_std/**",
      "**/model/**",
      "**/log*",
      "**/*.zip",
      "rerank_model_dir/rm.md",
      "repomix-output.*"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  }
}
````
