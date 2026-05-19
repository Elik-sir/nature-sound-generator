import src.config as cfg


def test_project_root_is_directory():
    assert cfg.PROJECT_ROOT.is_dir()


def test_raw_audio_dir_under_root():
    assert cfg.RAW_AUDIO_DIR == cfg.PROJECT_ROOT / "ESC-50-master" / "audio"


def test_num_classes():
    assert cfg.NUM_CLASSES == 50
