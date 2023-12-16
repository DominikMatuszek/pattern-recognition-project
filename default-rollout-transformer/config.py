CONFIG = {
    'teacher_model': 'deit3_large_patch16_224.fb_in1k',
    #'student_model': 'vit_base_patch16_224.mae',
    'student_model': 'deit_tiny_distilled_patch16_224',
    'student_out_dim': 1000,
    'dataset_path': 'val/',
    'epochs': 3,
    'batch_size': 32,
    'lr': 0.001,
    #'downsampling_strategy': 'MAE',
    'downsampling_strategy': 'DOWNSIZE',
    'downsampling_factor': 5, # Only used if downsampling_strategy is DOWNSIZE
}

if "mae" in CONFIG["student_model"] and CONFIG["downsampling_strategy"] == "DOWNSIZE":
    raise ValueError("Are you sure that you want to both downsize and use MAE?")

if CONFIG["downsampling_strategy"] == "MAE":
    del CONFIG["downsampling_factor"]