from dataclasses import dataclass, asdict, astuple
from typing import Final
import os
import argparse
import re
import json
from .utils import make_dir, remove_dir, sanitize_model_path

# directories
DN_DATA_ROOT: Final[str] = "data"

DN_MAPILLARY: Final[str] = 'mapillary'
DN_IMAGE_SEQ: Final[str] = 'image_seq'
DN_SFM_SEQ: Final[str] = 'SfM_seq'
DN_IMAGE_PANOMAP: Final[str] = 'image_panomap'

DN_GS: Final[str] = 'gs'

# files
FN_DICT_API_KEY: Final[str] = 'dict_api_key.json'

FN_DF_RECORD: Final[str] = "261021 crossing_accident_record(Form57).csv"
FN_DF_CROSSING: Final[str] = '261021 crossing_inventory_current(Form71).csv'

FN_DF_IMAGE_CAND: Final[str] = 'df_image_cand.csv'
FN_DF_IMAGE_IDS_PER_SEQ: Final[str] = 'df_image_ids_per_seq.csv'
FN_DF_IMAGE_SEQ: Final[str] = 'df_image_seq.csv'
FN_DF_IMAGE_PANOMAP: Final[str] = 'df_image_panomap.csv'

# configurations
TARGET_STATES: Final[tuple[str, ...]] = ('California', 'Texas', 'Nebraska', 'New York', 'Florida')

IMG_SEARCH_FIELDS: Final[tuple[str, ...]] = ("id",)
IMG_DETAIL_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "altitude", "computed_altitude", "geometry", "computed_geometry",  # geographic info
    "captured_at",  # timestamp
    "height", "width", "thumb_original_url",  # image
    "compass_angle", "computed_compass_angle", "computed_rotation", "exif_orientation", # orientation info
    "camera_type", "is_pano", "make", "model",  # camera info
    "sequence",  # sequence info
    "camera_parameters", "atomic_scale", "merge_cc", "mesh", "sfm_cluster", # SfM info
    "creator", "organization",  # uploader info
    "quality_score", # image quality info
    # "detections", # object detection - can be fetched using other API: https://www.mapillary.com/developer/api-documentation#detection
)

BBOX_OFFSET: Final[float] = 0.0001 # 0.00001 ≒ 1.11 meters
DIST_THRES_IMG_SEQ: Final[float] = 0.0005 # 0.00001 ≒ 1.11 meters

FOV: Final[int] = 85
BEARING_STEP: Final[int] = 45
PANOMAP_IMAGE_HW: Final[tuple[int, int]] = (1080, 1920)


def parse_args() -> argparse.Namespace:
    """Create an argument parser for building configs from the CLI and parse CLI args."""
    parser = argparse.ArgumentParser(description="Project configuration parser")

    args = parser.parse_args()
    
    return args


@dataclass(frozen=True)
class ScrapingConfig:
    target_states: tuple[str, ...]

    img_search_fields: tuple[str, ...]
    img_detail_fields: tuple[str, ...]
    bbox_offset: float
    dist_thres_img_seq: float

    def __post_init__(self):
        ### sanity check
        for state in self.target_states:
            assert state in ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District Of Columbia', 
                                     'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 
                                     'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 
                                     'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                                     'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                                     'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


@dataclass(frozen=True)
class GSConfig:
    fov: int
    bearing_step: int
    panomap_image_hw: tuple[int, int]


@dataclass(frozen=True)
class PathConfig:
    # generated directories
    dir_image_seq: str
    dir_SfM_seq: str
    dir_image_panomap: str
    dir_gs: str
    
    # files
    dict_api_key: str

    df_record: str
    df_crossing: str

    df_image_cand: str
    df_image_ids_per_seq: str
    df_image_seq: str
    df_image_panomap: str


@dataclass()
class APIkeyConfig:
    openai: str
    google: str
    textract: str
    mapillary: str


@dataclass(frozen=True)
class Config:
    scrp: ScrapingConfig
    gs: GSConfig
    path: PathConfig
    apikey: APIkeyConfig


def _compute_paths() -> PathConfig:
    dp_mapillary = os.path.join(DN_DATA_ROOT, DN_MAPILLARY)

    dp_image_seq = os.path.join(dp_mapillary, DN_IMAGE_SEQ)
    make_dir(dp_image_seq)
    dp_SfM_seq = os.path.join(dp_mapillary, DN_SFM_SEQ)
    make_dir(dp_SfM_seq)
    dp_image_panomap = os.path.join(dp_mapillary, DN_IMAGE_PANOMAP)
    make_dir(dp_image_panomap)
    dp_gs = os.path.join(DN_DATA_ROOT, DN_GS)
    make_dir(dp_gs)

    return PathConfig(
        dir_image_seq=dp_image_seq,
        dir_SfM_seq=dp_SfM_seq,
        dir_image_panomap=dp_image_panomap,
        dir_gs=dp_gs,

        dict_api_key=FN_DICT_API_KEY,

        df_record=os.path.join(DN_DATA_ROOT, FN_DF_RECORD),
        df_crossing=os.path.join(DN_DATA_ROOT, FN_DF_CROSSING),

        df_image_cand=os.path.join(dp_mapillary, FN_DF_IMAGE_CAND),
        df_image_ids_per_seq=os.path.join(dp_mapillary, FN_DF_IMAGE_IDS_PER_SEQ),
        df_image_seq=os.path.join(dp_image_seq, FN_DF_IMAGE_SEQ),
        df_image_panomap=os.path.join(dp_image_panomap, FN_DF_IMAGE_PANOMAP),
    )


def _load_api_key(path_cfg: PathConfig) -> APIkeyConfig:
    with open(path_cfg.dict_api_key, 'r') as f:
        dict_api_key = json.load(f)
    dict_api_key = {api: info['key'] for api, info in dict_api_key.items()}
    
    return APIkeyConfig(**dict_api_key)


def build_config(args_dict=None) -> Config:
    if args_dict is None:
        args = parse_args()
        args_dict = vars(args)
    scrp_cfg = ScrapingConfig(
        TARGET_STATES, IMG_SEARCH_FIELDS, IMG_DETAIL_FIELDS,
        BBOX_OFFSET, DIST_THRES_IMG_SEQ
    )
    gs_cfg = GSConfig(FOV, BEARING_STEP, PANOMAP_IMAGE_HW)
    path_cfg = _compute_paths()
    apikey_cfg = _load_api_key(path_cfg)
    return Config(scrp=scrp_cfg, gs=gs_cfg, path=path_cfg, apikey=apikey_cfg)


if __name__ == '__main__':
    cfg = build_config()