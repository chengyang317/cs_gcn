




def load_train_data(max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
    scene_graph_file = cfg.SCENE_GRAPH_FILE % \
        cfg.TRAIN.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
    data_reader = DataReader(
        imdb_file, shuffle=True, max_num=max_num,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        feature_type=cfg.FEAT_TYPE,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
        objects_max_num=cfg.W_FEAT,
        scene_graph_file=scene_graph_file,
        vocab_name_file=cfg.VOCAB_NAME_FILE,
        vocab_attr_file=cfg.VOCAB_ATTR_FILE,
        add_pos_enc=cfg.ADD_POS_ENC,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices