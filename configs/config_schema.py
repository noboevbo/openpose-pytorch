cfg_schema_augmentation = {
    'type': 'object',
    'schema': {
        'target_dist': {'type': 'float'},
        'scale_prob': {'type': 'integer'},
        'scale_min': {'type': 'float'},
        'scale_max': {'type': 'float'},
        'max_rotate_degree': {'type': 'integer'},
        'center_perterb_max': {'type': 'integer'},
        'flip_prob': {'type': 'float'},
        'sigma': {'type': 'integer'},
    }
}

cfg_schema_convert_caffe = {
    'type': 'object',
    'schema': {
        'caffe_model': {'type': 'string'},
        'deploy_model': {'type': 'string'},
        'pytorch_model': {'type': 'string'},
        'test_image': {'type': 'string'},
    }
}

cfg_schema_general = {
    'type': 'object',
    'schema': {
        'input_width': {'type': 'integer'},
        'input_height': {'type': 'integer'},
        'stride': {'type': 'integer'},
        'debug_timers': { 'type': 'boolean'},
        'additional_debug_timers': { 'type': 'boolean'}
    }
}

cfg_schema_convert = {
    'type': 'object',
    'schema': {
        'caffe': cfg_schema_convert_caffe
    }
}

cfg_schema_training = {
                'type': 'object',
                'schema': {
                    'weight_dir': {'type': 'string'},
                    'trained_model_dir': {'type': 'string'},
                    'log_dir': {'type': 'string'},
                    'checkpoint_model_base_dir' : {'type': 'string'},
                    'checkpoint_model_path': {'type': 'string', 'nullable': True},
                    'checkpoint_epoch': {'type': 'integer'},
                    'checkpoint_best_model_loss': {'type': 'float', 'nullable': True},

                    'learning_rate': {'type': 'float'},  # Base learning rate, is changed per layer grp
                    'batch_size': {'type': 'integer'},
                    'gamma': {'type': 'float'},  # 4 Learning Rate Scheduler
                    'stepsize': {'type': 'integer'},
                # in original code each epoch is 121746 and step change is on 17th epoch

                    'momentum': {'type': 'float'},
                    'weight_decay': {'type': 'float'},
                    'augmentation': cfg_schema_augmentation
                }
            }

cfg_schema_network = {
    'type': 'object',
    'schema': {
        'use_gpu': {'type': 'integer'},
        'gpu_device_number': {'type': 'integer'},
        'model_state_file': {'type': 'string'},
        'scale_search': {
            'type': 'list',
            'schema': {
                'type': 'float'
            }
        },
        'pad_color': {'type': 'integer'},
        'heatmap_thresh': {'type': 'float'},
        'limb_num_samples': {'type': 'integer'},
        'limb_sample_score_thresh': {'type': 'float'},
        'limb_samples_over_thresh': {'type': 'float'},
        'skeleton_min_limbs': {'type': 'integer'},
        'skeleton_limb_score': {'type': 'float'},
        'stage_delay_epochs': {
            'type': 'list',
            'schema': {
                'type': 'integer'
            }
        }
    }
}

cfg_schema_dataset = {
    'type': 'object',
    'schema': {
        'base_dir': {'type': 'string'},
        'train_annotation_dir': {'type': 'string'},
        'train_img_dir': {'type': 'string'},
        'train_convert_hdf5': {'type': 'string'},
        'train_hdf5': {'type': 'string'},
        'val_annotation_dir': {'type': 'string'},
        'val_img_dir': {'type': 'string'},
        'val_convert_hdf5': {'type': 'string'},
        'val_hdf5': {'type': 'string'},
    }
}

cfg_schema = {
    'general': cfg_schema_general,
    'convert': cfg_schema_convert,
    'network': cfg_schema_network,
    'train': cfg_schema_training,
    'dataset': cfg_schema_dataset
}
