from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import requests
import py360convert
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import pathlib
from .config import Config
from .utils import (as_int, as_float, remove_dir, make_dir, make_dirs,
                    prepare_df_crossing,
                    prepare_df_image_cand, prepare_df_image_ids_per_seq, prepare_df_image_seq,
                    prepare_df_image_panomap)
from pprint import pprint
from PIL import Image
import math
import py360convert
import math
import zlib
import json


IMG_POS_ZFILL = 5


class MapillaryAPIClient:
    def __init__(self, cfg):
        self.api_key = cfg.apikey.mapillary
        self.img_search_fields = ','.join(cfg.scrp.img_search_fields)
        self.img_detail_fields = ','.join(cfg.scrp.img_detail_fields)
        
    def request_images(self, bbox: str):
        """
        Query Mapillary for images inside a bounding box.
        Returns a list of image objects with basic metadata.
        """
        url = (
            "https://graph.mapillary.com/images"
            f"?access_token={self.api_key}"
            f"&bbox={bbox}"
            f"&fields={self.img_search_fields}"
            # f"&limit={limit}"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        resp_json = resp.json()

        # Mapillary returns {"data": [ ...images... ], ...maybe paging...}
        data = resp_json.get("data", [])
        if len(data) == 2000:
            raise ValueError(f"Warning: Mapillary returned 2000 (max limit) images for bbox {bbox}. Some images may be missing. Consider splitting the bbox into smaller areas.")
        return data

            
    def request_image_details(self, image_id: str):
        """
        Ask Mapillary for richer metadata for one specific image.
        Returns a dict with fields we asked for, including thumb_1024_url.
        """
        url = (
            f"https://graph.mapillary.com/{image_id}"
            f"?access_token={self.api_key}"
            f"&fields={self.img_detail_fields}"
        )

        try:
            resp = requests.get(url)
            resp.raise_for_status()
        except:
            url = url.replace(',sfm_cluster', '')
            resp = requests.get(url)
            resp.raise_for_status()
        return resp.json()

    def request_image_seq(self, seq_id: str):
        """
        Query Mapillary for image sequence with a sequence key.
        Returns a list of image objects with basic metadata.
        """
        url = (
            'https://graph.mapillary.com/image_ids'
            f"?access_token={self.api_key}"
            f'&sequence_id={seq_id}'
        )

        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()


class MapillaryImageFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mapillary_client = MapillaryAPIClient(cfg)
        self.img_cols = self.__set_col_list()

    def __set_col_list(self):
        """
        Set up the columns for the image dataframes
        """
        img_cols = list(self.cfg.scrp.img_detail_fields)
        # img_cols.remove('camera_parameters')
        # img_cols.extend(['focal_length', 'k1', 'k2'])
        img_cols.remove('geometry')
        img_cols.remove('computed_geometry')
        img_cols.extend(['img_lon', 'img_lat', 'computed_img_lon', 'computed_img_lat', 'dist', 'computed_dist'])
        rename_map = {'id': 'img_id', 'sequence': 'seq_id'}
        img_cols = [rename_map.get(col, col) for col in img_cols]
        img_cols = ['crossing_id'] + img_cols        
        return img_cols
    
    def load_df_image_cand(self):
        if os.path.exists(self.cfg.path.df_image_cand):
            df_image_cand = prepare_df_image_cand(self.cfg)
        else:
            df_image_cand = pd.DataFrame(columns=self.img_cols)
            df_image_cand.to_csv(self.cfg.path.df_image_cand, index=False)
        return df_image_cand
    
    def load_df_image_ids_per_seq(self):
        if os.path.exists(self.cfg.path.df_image_ids_per_seq):
            df_image_ids_per_seq = prepare_df_image_ids_per_seq(self.cfg)
        else:
            df_image_ids_per_seq = pd.DataFrame(columns=['seq_id', 'img_ids'])
            df_image_ids_per_seq.to_csv(self.cfg.path.df_image_ids_per_seq, index=False)
        return df_image_ids_per_seq
    
    def load_df_image_seq(self):
        if os.path.exists(self.cfg.path.df_image_seq):
            df_image_seq = prepare_df_image_seq(self.cfg)
        else:
            df_image_seq = pd.DataFrame(columns=self.img_cols + ['img_pos'])
            df_image_seq.to_csv(self.cfg.path.df_image_seq, index=False)
        return df_image_seq
    
    def fetch_image_cands_per_crossing(self) -> pd.DataFrame:
        df_crossing = prepare_df_crossing(self.cfg)
        df_image_cand = self.load_df_image_cand()
        for i, row in tqdm(df_crossing[['crossingid', 'latitude', 'longitude']].iterrows(), total=df_crossing.shape[0]):
            crossing_id, xing_lat, xing_lon = row
            if crossing_id in df_image_cand['crossing_id'].values:
                continue
            bbox_exact_match = f"{xing_lon - self.cfg.scrp.bbox_offset},{xing_lat - self.cfg.scrp.bbox_offset},{xing_lon + self.cfg.scrp.bbox_offset},{xing_lat + self.cfg.scrp.bbox_offset}"
            imgs = self.mapillary_client.request_images(bbox_exact_match)

            details_concat = [{'crossing_id': crossing_id}]
            for img in imgs:
                img_id = img["id"]
                if img_id in df_image_cand['img_id'].values:
                    continue
                details = self.mapillary_client.request_image_details(img_id)
                details['crossing_id'] = crossing_id
                details = self.reformat_image_details(details, xing_lat, xing_lon)
                details_concat.append(details)
            
            df_image_temp = pd.DataFrame(details_concat, columns=df_image_cand.columns)
            df_image_cand = pd.concat([df_image_cand, df_image_temp], ignore_index=True)
            
            if i % 10 == 0: # type: ignore
                df_image_cand.to_csv(self.cfg.path.df_image_cand, index=False)
            
        df_image_cand.to_csv(self.cfg.path.df_image_cand, index=False) # finally save the complete df

        return df_image_cand
    
    def fetch_image_ids_per_seq(self) -> pd.DataFrame:
        df_image_cand = self.load_df_image_cand()
        df_image_ids_per_seq = self.load_df_image_ids_per_seq()

        df_image_cand = df_image_cand.dropna(subset=['seq_id'])

        # First get all the image sequences for the crossings
        for i, seq_id in tqdm(enumerate(df_image_cand['seq_id'].unique()), total=df_image_cand['seq_id'].nunique()):
            if seq_id in df_image_ids_per_seq['seq_id'].unique():
                continue
            
            resp = self.mapillary_client.request_image_seq(seq_id)
            seq = resp['data']
            if len(resp.keys()) != 1:
                raise ValueError(f"Unexpected response format (another key other than 'data') for sequence {seq_id}: {resp}")
            if len(seq) == 0:
                raise ValueError(f"seq length is 0 for sequence {seq_id}. This may indicate that the sequence has no images or that the sequence ID is invalid.")

            data = [{'seq_id': seq_id, 'img_ids': [resp_img_id['id'] for resp_img_id in seq]}]
            df_image_ids_per_seq_temp = pd.DataFrame(data)
            df_image_ids_per_seq = pd.concat([df_image_ids_per_seq, df_image_ids_per_seq_temp], ignore_index=True)

            if i % 10 == 0:
                df_image_ids_per_seq.to_csv(self.cfg.path.df_image_ids_per_seq, index=False)
        df_image_ids_per_seq.to_csv(self.cfg.path.df_image_ids_per_seq, index=False)

        return df_image_ids_per_seq

    def fetch_image_seqs(self) -> pd.DataFrame:
        df_crossing = prepare_df_crossing(self.cfg)
        df_image_cand = self.load_df_image_cand()
        df_image_ids_per_seq = self.load_df_image_ids_per_seq()
        df_image_seq = self.load_df_image_seq()

        df_image_cand = df_image_cand.dropna(subset=['seq_id'])
        df_image_gp = df_image_cand.groupby(['crossing_id', 'seq_id'])
        df_crossing_seq = df_image_gp.apply(lambda x: x.loc[x['dist'].idxmin()]).reset_index(drop=False) # for each crossing and sequence group, leave only an image with the minimum distance to the crossing, which is min of 'dist' column.

        df_image_ids_per_seq = df_image_ids_per_seq.set_index('seq_id')
        assert df_image_ids_per_seq['img_ids'].str.len().max() <= 10**IMG_POS_ZFILL, \
            f"The image position index (img_pos) will exceed the maximum number of digits allowed by IMG_POS_ZFILL={IMG_POS_ZFILL}. Please increase IMG_POS_ZFILL to accommodate the maximum sequence length."

        for i, row in tqdm(df_crossing_seq.iterrows(), total=df_crossing_seq.shape[0]):
            crossing_id = row['crossing_id']
            seq_id = row['seq_id']
            img_id = row['img_id']
            xing_lat, xing_lon = df_crossing.loc[df_crossing['CROSSING'] == crossing_id, ['LATITUDE', 'LONGITUD']].values[0]
            if not df_image_seq[(df_image_seq['crossing_id'] == crossing_id) & (df_image_seq['seq_id'] == seq_id)].empty:
                continue

            img_ids = df_image_ids_per_seq.loc[seq_id, 'img_ids']
            try:
                img_pos = img_ids.index(str(img_id)) # type: ignore
            except:
                img_pos = float('nan')
                print(f'Warning: img_id {img_id} not found in the image sequence {seq_id} for crossing {crossing_id}. This may indicate that there is a mismatch.')

            row['img_pos'] = img_pos
            df_image_seq_temp = pd.DataFrame(columns=df_image_seq.columns)
            df_image_seq_temp = pd.concat([df_image_seq_temp, row.to_frame().T])
            if pd.isna(img_pos):
                df_image_seq = pd.concat([df_image_seq, df_image_seq_temp], ignore_index=True)
                continue

            search_config = {
                'forward': {'within_boundary': True, 'pos_step': 1},
                'backward': {'within_boundary': True, 'pos_step': -1}
            }
            for search_direction, config in search_config.items():
                img_pos_temp = img_pos
                while config['within_boundary']:
                    img_pos_temp += config['pos_step']
                    if img_pos_temp < 0 or img_pos_temp >= len(img_ids): # type: ignore
                        break
                    img_id_temp = img_ids[img_pos_temp] # type: ignore
                    details = self.mapillary_client.request_image_details(img_id_temp) # type: ignore
                    details['crossing_id'] = crossing_id
                    details = self.reformat_image_details(details, xing_lat, xing_lon)
                    details['img_pos'] = img_pos_temp

                    if details['dist'] > self.cfg.scrp.dist_thres_filter_img_seq:
                        config['within_boundary'] = False
                    df_details_temp = pd.DataFrame([details], columns=df_image_seq.columns)
                    if search_direction == 'forward':
                        df_image_seq_temp = pd.concat([df_image_seq_temp, df_details_temp], ignore_index=True)
                    elif search_direction == 'backward':
                        df_image_seq_temp = pd.concat([df_details_temp, df_image_seq_temp], ignore_index=True)
                    else:
                        raise ValueError(f"Unexpected search_direction: {search_direction}. Expected 'forward' or 'backward'.")
            
            df_image_seq = pd.concat([df_image_seq, df_image_seq_temp], ignore_index=True)
            if i % 10 == 0: # type: ignore
                df_image_seq.to_csv(self.cfg.path.df_image_seq, index=False)
        df_image_seq.to_csv(self.cfg.path.df_image_seq, index=False)

        return df_image_seq
    
    def download_image(self, url: str, out_path: pathlib.Path):
        """
        Download the content and save it locally.
        """
        resp = requests.get(url)
        resp.raise_for_status()

        out_path.write_bytes(resp.content)

    def download_image_seqs(self):
        df_image_seq = self.load_df_image_seq()

        # create directories for each crossing and sequence
        df_image_dir_name = df_image_seq.drop_duplicates(subset=['crossing_id', 'seq_id'], keep='first')[['crossing_id', 'seq_id']]
        make_dirs(df_image_dir_name, self.cfg.path.dir_image_seq)

        # start downloading images
        all_downloaded = False
        while not all_downloaded:
            try:
                for i, row in tqdm(df_image_seq.iterrows(), total=df_image_seq.shape[0]):
                    crossing_id = row['crossing_id']
                    seq_id = row['seq_id']
                    img_id = as_int(row['img_id'])
                    img_pos = str(as_int(row['img_pos'])).zfill(IMG_POS_ZFILL)
                    thumb_url = row["thumb_original_url"]
                    if pd.isna(img_id):
                        continue
                    fp_output = pathlib.Path(os.path.join(self.cfg.path.dir_image_seq, crossing_id, seq_id, f"{img_pos}_{img_id}.jpg"))
                    if not fp_output.exists() and pd.notna(thumb_url):
                        self.download_image(thumb_url, fp_output)
                all_downloaded = True
            except:
                time_to_sleep = 10
                print(f"An error occurred during the download process. Retrying in {time_to_sleep} seconds...")
                time.sleep(time_to_sleep)
                continue

    def reformat_image_details(self, details, xing_lat, xing_lon):
        img_id = details.pop('id')
        details['img_id'] = img_id
        seq_id = details.pop('sequence')
        details['seq_id'] = seq_id
        if details.get('geometry', None):
            assert details['geometry']['type'] == 'Point'
            geometry = details.pop('geometry')
            details['img_lon'] = geometry['coordinates'][0]
            details['img_lat'] = geometry['coordinates'][1]
            dist = ((xing_lat - details['img_lat'])**2 + (xing_lon - details['img_lon'])**2)**0.5
            details['dist'] = dist
            # assert dist <= self.cfg.scrp.bbox_offset * 2**0.5
        if details.get('computed_geometry', None):
            assert details['computed_geometry']['type'] == 'Point'
            computed_geometry = details.pop('computed_geometry')
            details['computed_img_lon'] = computed_geometry['coordinates'][0]
            details['computed_img_lat'] = computed_geometry['coordinates'][1]
            computed_dist = ((xing_lat - details['computed_img_lat'])**2 + (xing_lon - details['computed_img_lon'])**2)**0.5
            details['computed_dist'] = computed_dist
            # assert computed_dist <= cfg.scrp.bbox_offset * 2**0.5
        if details.get('computed_rotation', None):
            assert isinstance(details['computed_rotation'], list)
        if details.get('captured_at', None):
            captured_at = details.pop('captured_at')
            details['captured_at'] = pd.to_datetime(captured_at, unit='ms')
        # if details.get('camera_parameters', None):
        #     camera_parameters = details.pop('camera_parameters')
        #     assert len(camera_parameters) == 3
        #     details['focal_length'] = camera_parameters[0]
        #     details['k1'] = camera_parameters[1]
        #     details['k2'] = camera_parameters[2]
        # else:
        #     details['focal_length'] = None
        #     details['k1'] = None
        #     details['k2'] = None
        # print(f"computed:\t{computed_dist :.6f}")
        # print(f"actual:\t{dist :.6f}")
        # pprint(details, sort_dicts=False)
        
        return details
    
    def extract_view(self, e_img: np.ndarray, h_fov=90, yaw_deg=0, pitch_deg=0, out_hw=(512, 512)) -> np.ndarray:
        """
        Take a perspective view from the equirectangular pano.

        yaw_deg  (u_deg in py360convert): -left / +right (0 = forward)
        pitch_deg (v_deg): -down / +up
        fov_deg: horizontal FOV (or (h_fov, v_fov) tuple)
        out_hw: (height, width) of output image
        """
        aspect = out_hw[1] / out_hw[0]
        v_fov = 2 * math.degrees(math.atan(math.tan(math.radians(h_fov/2)) / aspect))
        pers = py360convert.e2p(
            e_img=e_img,
            fov_deg=(h_fov, v_fov),
            u_deg=yaw_deg,       # left/right
            v_deg=pitch_deg,     # up/down
            out_hw=out_hw,       # output size (H, W)
            in_rot_deg=0,
            mode="bilinear"
        )
        return pers


class MapillarySfMFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
            
    def download_SfM(self, sfm_url: str, out_path: pathlib.Path):
        """
        Download the SfM data from Mapillary and save it as a JSON file.
        """
        resp = requests.get(sfm_url)
        resp.raise_for_status()

        bin_data = resp.content
        json_str = zlib.decompress(bin_data)  # data is compressed with zlib
        json_SfM = json.loads(json_str)
        
        out_path.write_text(json.dumps(json_SfM, indent=4))

    def fetch_SfM_per_seq(self):
        df_image_seq = prepare_df_image_seq(self.cfg)
        df_image_seq = df_image_seq.dropna(subset=['sfm_cluster'])
        assert df_image_seq['sfm_cluster'].apply(lambda x: len(x)).max() <= 2
        df_image_seq['sfm_id'] = df_image_seq['sfm_cluster'].apply(lambda x: x['id'])
        df_image_seq['sfm_url'] = df_image_seq['sfm_cluster'].apply(lambda x: x['url'] if 'url' in x else float('nan'))
        df_sfm_seq = df_image_seq[['crossing_id', 'seq_id', 'sfm_id', 'sfm_url']]
        df_sfm_seq = df_sfm_seq.dropna(subset=['sfm_url'])
        df_sfm_seq = df_sfm_seq.drop_duplicates(subset=['crossing_id', 'seq_id', 'sfm_id'])

        # create dirs
        df_sfm_seq_dir_name = df_sfm_seq.drop_duplicates(subset=['crossing_id', 'seq_id'], keep='first')[['crossing_id', 'seq_id']]
        make_dirs(df_sfm_seq_dir_name, self.cfg.path.dir_SfM_seq)
        
        # start downloading SfM zlib files
        for i, row in tqdm(df_sfm_seq.iterrows(), total=df_sfm_seq.shape[0]):
            crossing_id = row['crossing_id']
            seq_id = row['seq_id']
            sfm_id = row['sfm_id']
            sfm_url = row['sfm_url']
            if pd.isna(sfm_url):
                continue
            fp_output = pathlib.Path(os.path.join(self.cfg.path.dir_SfM_seq, crossing_id, seq_id, f"{sfm_id}.json"))
            if not fp_output.exists():
                self.download_SfM(sfm_url, fp_output)


class MapillarySfMExaminer:        
    camera_fields = {
        'perspective': ['projection_type', 'width', 'height', 'focal', 'k1', 'k2'],
        'fisheye': ['projection_type', 'width', 'height', 'focal', 'k1', 'k2'],
        'spherical': ['projection_type', 'width', 'height'],
        'equirectangular': ['projection_type', 'width', 'height'],
        'fisheye62': ['projection_type', 'width', 'height', 'focal_x', 'focal_y', 'c_x', 'c_y', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2'],
        'brown': ['projection_type', 'width', 'height', 'focal_x', 'focal_y', 'c_x', 'c_y', 'k1', 'k2', 'k3', 'p1', 'p2']
    }

    shot_fields = {
        1: ['rotation', 'translation', 'camera', 'orientation', 'capture_time', 'gps_dop', 'gps_position', 'compass', 'skey', 'vertices', 'faces', 'scale', 'covariance', 'merge_cc'],
        2: ['rotation', 'translation', 'camera', 'orientation', 'capture_time', 'gps_dop', 'gps_position', 'accelerometer', 'compass', 'skey', 'scale', 'merge_cc'],
    }

    point_fields = ['color', 'coordinates']

    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def assert_camera_fields(self, cameras):
        for camera_id, camera in cameras.items():
            projection_type = camera['projection_type']
            if projection_type in self.camera_fields:
                assert set(camera.keys()) == set(self.camera_fields[projection_type]), f"Camera has unexpected fields: {camera.keys()} vs {self.camera_fields[projection_type]}"
            else:
                raise ValueError(f"Unknown projection type {projection_type} for camera {camera}")

    def assert_shot_fields(self, shots):
        raise NotImplementedError("assert_shot_fields is not fully implemented yet")
        field_types = []
        for shot_id, shot in shots.items():
            if set(shot.keys()) == set(self.shot_fields[1]):
                field_types.append(1)
            elif set(shot.keys()) == set(self.shot_fields[2]):
                field_types.append(2)
            elif 'compass' not in shot and 'skey' not in shot:
                print()
                print(f"Shot missing 'compass' and 'skey': {shot}")
                continue
            elif 'accelerometer' not in shot and 'compass' not in shot and 'scale' not in shot and 'merge_cc' not in shot:
                print()
                print(f"Shot missing 'accelerometer', 'compass', 'scale', and 'merge_cc': {shot}")
                continue
            else:
                raise ValueError(f"Shot has unexpected fields: {shot.keys()} vs {self.shot_fields[1]} or {self.shot_fields[2]}")
            
            if len(set(field_types)) > 1:
                raise ValueError(f"Shots have mixed field types: {field_types}")
            
            assert set(shot['compass']) == {'angle', 'accuracy'}, f"Compass field has unexpected keys: {shot['compass'].keys()}"
        
        df_shots = pd.DataFrame.from_dict(shots, orient='index').reset_index(names='shot_id')
        assert (df_shots[['rotation', 'translation']].map(len) == 3).all().all(), "Rotation or translation field does not have length 3 for all shots"
        # assert df_shots['camera'].nunique() == 1, "Expected all shots to have the same camera"
        assert df_shots['orientation'].nunique() == 1, "Expected all shots to have the same orientation"
        # assert df_shots['gps_dop'].nunique() == 1, "Expected all shots to have the same gps_dop"
        assert (df_shots['gps_position'].str.len() == 3).all(), "Expected all shots to have gps_position of length 3"
        # assert df_shots['skey'].nunique() == 1, "Expected all shots to have the same skey"
        if field_types.count(1) > 0 and field_types.count(2) == 0:
            assert (df_shots[['vertices', 'faces', 'covariance']] == [[], [], []]).all().all(), "Expected all shots to have empty vertices, faces, and covariance"
        # df_shots['scale']
        # df_shots['merge_cc']

    def assert_point_fields(self, points):
        if points == {}:
            pass
        else:
            df_points = pd.DataFrame.from_dict(points, orient='index').reset_index(names='point_id')
            assert (df_points[self.point_fields].map(len) == 3).all().all(), "Expected all points to have color and coordinates of length 3"


def fetch_image_cand(cfg: Config) -> pd.DataFrame:
    image_fetcher = MapillaryImageFetcher(cfg)
    df_image_cand = image_fetcher.fetch_image_cands_per_crossing()
    print(f"Fetched {len(df_image_cand)} images within bounding boxes around {df_image_cand['crossing_id'].nunique()} crossings.")
    return df_image_cand


def fetch_image_seq(cfg: Config, download=False) -> pd.DataFrame:
    image_fetcher = MapillaryImageFetcher(cfg)
    df_image_ids_per_seq = image_fetcher.fetch_image_ids_per_seq()
    print(f"Fetched an image-id list for each image sequence. Total sequences: {len(df_image_ids_per_seq)}")
    df_image_seq = image_fetcher.fetch_image_seqs()
    print(f"Fetched {len(df_image_seq)} detail information for all images.")
    if download:
        image_fetcher.download_image_seqs()
        print(f"Downloaded all images for each crossing and sequence.")
    return df_image_seq


def fetch_SfM(cfg: Config):
    sfm_fetcher = MapillarySfMFetcher(cfg)
    sfm_fetcher.fetch_SfM_per_seq()
    print(f"Downloaded SfM data for all sequences.")


def examine_SfM(cfg: Config):
    sfm_examiner = MapillarySfMExaminer(cfg)
    STARTING_POINT = 0
    list_crossing = os.listdir(cfg.path.dir_SfM_seq)
    for i, crossing in enumerate(tqdm(list_crossing, desc="Crossings")):
        if i < STARTING_POINT:
            continue
        list_seq = os.listdir(os.path.join(cfg.path.dir_SfM_seq, crossing))
        for seq in tqdm(list_seq, desc="Sequences", leave=False):
            list_SfM = os.listdir(os.path.join(cfg.path.dir_SfM_seq, crossing, seq))
            for SfM in tqdm(list_SfM, desc="SfM Clusters", leave=False):
                path_SfM_json = os.path.join(cfg.path.dir_SfM_seq, crossing, seq, SfM)
                with open(path_SfM_json, 'r') as f:
                    SfM_json = json.load(f)
                    assert len(SfM_json) == 1, f"Expected 1 SfM cluster, but got {len(SfM_json)}"
                    SfM_data = SfM_json[0]
                    cameras = SfM_data['cameras']
                    shots = SfM_data['shots']
                    points = SfM_data['points']
                    
                    sfm_examiner.assert_camera_fields(cameras)
                    # sfm_examiner.assert_shot_fields(shots)
                    sfm_examiner.assert_point_fields(points)


def visualize_SfM(cfg: Config, crossing_id, seq_id, sfm_id):
    # df_image_seq = prepare_df_image_seq(cfg)
    # df_image_seq['sfm_id'] = df_image_seq['sfm_cluster'].apply(lambda x: x['id'] if pd.notna(x) else float('nan'))
    # df_sfm = df_image_seq[['crossing_id', 'seq_id', 'sfm_id', 'img_lon', 'img_lat']]
    # df_sfm = df_sfm.dropna(subset=['sfm_id', 'seq_id'], how='any')
    # df_sfm = df_sfm.drop_duplicates(subset=['crossing_id', 'seq_id', 'sfm_id'])

    # crossing_id, seq_id, sfm_id, img_lon, img_lat = df_sfm.sample(1).iloc[0]
    path_SfM_json = os.path.join(cfg.path.dir_SfM_seq, crossing_id, seq_id, f'{str(int(sfm_id))}.json')
    if not os.path.exists(path_SfM_json):
        raise FileNotFoundError(f"SfM JSON file not found: {path_SfM_json}")
    with open(path_SfM_json, 'r') as f:
        SfM_json = json.load(f)
        assert len(SfM_json) == 1, f"Expected 1 SfM cluster, but got {len(SfM_json)}"
    SfM_data = SfM_json[0]

    # cameras = SfM_data['cameras']
    # shots = SfM_data['shots']
    points = SfM_data['points']
    # points = {f'{sfm_id}_{point_id}': point for point_id, point in points.items()}
    if len(points) < 10000:
        raise ValueError(f"Insufficient points in SfM cluster: {len(points)}")

    sfm_visualizer = MapillarySfMVisualizer(cfg, points)
    sfm_visualizer.visualize()


def __to_be_used():
    raise NotImplementedError("The function `_get_image_seq_detail` is not yet implemented. It should fetch detailed information for each image in the sequences, but this functionality is currently a placeholder.")
    ############### filtering df_image
    df_image = df_image.dropna(subset=['seq_id'])
    df_image['img_id'] = df_image['img_id'].astype(int)
    # df_image = df_image[df_image['is_pano'] == 1]
    # df_image = df_image[df_image['camera_type'] == 'spherical'] # [spherical, equirectangular] equirectangular images require diff view extraction mechanism
    
    ############### filtering df_crossing
    target_crossings = df_image['crossing_id'].unique()
    df_crossing = df_crossing[df_crossing['CROSSING'].isin(target_crossings)]
    df_crossing = df_crossing[df_crossing['LLSOURCE'].isin(['1'])] # ['1', '2', ' '];  ' ' mostly incorrect, '2' sometimes incorrect
    df_crossing = df_crossing[df_crossing['XPURPOSE'] == 1] # 1: highway, 2: pedestrian pathway, 3: train station / [2,3] images are generally not available
    df_crossing = df_crossing.drop(['HIGHWAY', 'RRDIV', 'RRSUBDIV'], axis=1)

    ############### merge
    df_image = df_image.merge(df_crossing[['CROSSING', 'LATITUDE', 'LONGITUD', 'STREET']], left_on='crossing_id', right_on='CROSSING')
    df_image = df_image.drop(columns=['CROSSING'])
    df_image = df_image.rename(columns={'LATITUDE': 'crossing_lat', 'LONGITUD': 'crossing_lon', 'STREET': 'crossing_street', 'lat': 'img_lat', 'lon': 'img_lon'})

    # # check how many image sequences span across multiple crossings
    # df_drop_dup = df_image.drop_duplicates(subset=['crossing_id', 'seq_id'])
    # seq_spanning = df_drop_dup['seq_id'].duplicated().sum()
    # print(f"{seq_spanning} / {df_drop_dup.shape[0]} sequences span across multiple crossings.")


def tentative():
    ############### using only images with distance over the threshold
    cols_df_min_dist = ['crossing_id', 'img_id', 'captured_at', 'compass_angle', 'computed_compass_angle', 'sequence', 'img_lat', 'img_lon', 'crossing_lat', 'crossing_lon', 'dist']
    df_min_dist = df_image.loc[df_image.groupby("crossing_id")["dist"].idxmin()][cols_df_min_dist].reset_index(drop=True)
    df_min_dist = df_min_dist[df_min_dist['dist'] <= cfg.scrp.dist_thres_filter_img] # it seems actual GPS location is more accurate than computed GPS location, so I only use `dist` here, not `computed_dist`.
    
    ############### get image seq
    (((df_image['lat'] - df_image['computed_lat'])**2 + (df_image['lon'] - df_image['computed_lon'])**2)**0.5).dropna().sort_values()
    for i, row in tqdm(df_min_dist.iterrows(), total=df_min_dist.shape[0]):
        crossing_id = row['crossing_id']
        if crossing_id in df_image_seq['crossing_id'].values:
            continue
        seq_id = row['sequence']
        xing_lat, xing_lon = row[['crossing_lat', 'crossing_lon']]
        
        df_seq_temp = df_image[(df_image['crossing_id'] == crossing_id) & (df_image['sequence'] == seq_id)]
        df_seq_temp = df_seq_temp[(df_seq_temp['computed_compass_angle'].notna())]
        # select_img_within = [0.0001, 0.0002]
        # df_seq_temp = df_seq_temp[(select_img_within[0] <= df_seq_temp['dist']) & (df_seq_temp['dist'] <= select_img_within[1])]
        
        if df_seq_temp.shape[0] <= 1:
            continue
        
        images = mapillary_image_client.request_image_seq(seq_id)['data']
        images = [image['id'] for image in images]
        # for image in tqdm(images, leave=False):
        #     img_id = image['id']
        #     details = mapillary_image_client.request_image_details(img_id)
        #     dist = ((row[['crossing_lon', 'crossing_lat']] - details['geometry']['coordinates'])**2).sum()**0.5
        #     if dist > cfg.scrp.bbox_offset * 2 or dist < cfg.scrp.bbox_offset / 2:
        #         continue
        #     assert details['geometry']['type'] == 'Point'
        #     geometry = details.pop('geometry')
        #     details['lon'] = geometry['coordinates'][0]
        #     details['lat'] = geometry['coordinates'][1]
        
        # print(df_seq_temp)
        df_image_seq_temp = pd.DataFrame(columns=df_image_seq.columns)
        for _, seq_row in df_seq_temp.iterrows():
            img_id = str(int(seq_row['img_id']))
            if img_id not in images or pd.isna(seq_row['computed_compass_angle']):
                continue
            img_pos = str(images.index(str(img_id))).zfill(4)
            lat, lon = seq_row[['lat', 'lon']]
            # camera_yaw_deg = seq_row['compass_angle'] # from the raw GPS trajectory possibly with error away from the actual roadway
            camera_yaw_deg = seq_row['computed_compass_angle'] # corrected version according to the actual roadway
            # _, camera_pitch_deg, camera_yaw_deg = list(map(lambda d: int((360 + math.degrees(d)) % 360), seq_row['computed_rotation'])) # from rotation [roll, pitch, yaw]

            lat1, lon1, lat2, lon2 = map(math.radians, [lat, lon, xing_lat, xing_lon])
            d_lon = lon2 - lon1
            x = math.sin(d_lon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
            bearing = math.degrees(math.atan2(x, y))
            bearing = int((bearing - camera_yaw_deg + 360) % 360)
            
            fp_img = os.path.join(cfg.path.dir_scraped_images, f'{img_id}.jpg')
            if not os.path.exists(fp_img):
                continue
            img = np.array(Image.open(fp_img).convert("RGB"))
            view = mapillary_image_client.extract_view(img, h_fov=90, yaw_deg=bearing, pitch_deg=0, out_hw=(720, 960))
            out_img = Image.fromarray(view)
            dp_crossing_seq = os.path.join(cfg.path.dir_image_seq, crossing_id, seq_id)
            make_dir(dp_crossing_seq)
            fp_img_seq = os.path.join(dp_crossing_seq, f'{img_pos}_{img_id}.jpg')
            if not os.path.exists(fp_img_seq):
                out_img.save(fp_img_seq)

            df_image_seq_temp.loc[len(df_image_seq_temp)] = [crossing_id, seq_id, img_pos, img_id, bearing]
        
        if not df_image_seq_temp.empty:
            df_image_seq = pd.concat([df_image_seq, df_image_seq_temp])
        
        if i % 10 == 0: # type: ignore
            df_image_seq.to_csv(cfg.path.df_image_seq, index=False)

    df_image_seq.to_csv(cfg.path.df_image_seq, index=False)
    return df_image_seq


def scrape_3D(cfg: Config) -> pd.DataFrame:
    mapillary_image_client = MapillaryImageFetcher(cfg)
    columns_df_3D = ['crossing_id', 'seq_id', 'img_pos', 'img_id', 'bearing', 'captured_at', 'dist', 'atomic_scale', 'merge_cc']
    columns_df_3D_add = ['sfm_id', 'sfm_url', 'mesh_id', 'mesh_url']

    if os.path.exists(cfg.path.df_3D):
        df_3D = prepare_df_3D(cfg)
    else:
        df_3D = pd.DataFrame(columns=columns_df_3D + columns_df_3D_add)
    
    df_image = prepare_df_image(cfg)
    df_image = df_image.dropna(subset=['img_id'])
    df_image_seq = prepare_df_image_seq(cfg)

    df_merge_temp = df_image_seq.merge(df_image, left_on=['crossing_id', 'img_id'], right_on=['crossing_id', 'img_id'], how='left').drop(columns=['crossing_id', 'img_id', 'sequence']).copy(deep=True)
    df_merge_temp = df_merge_temp[~df_merge_temp['img_id'].isin(df_3D['img_id'].values)]
    df_merge_temp = df_merge_temp[columns_df_3D].copy(deep=True)
    df_merge_temp[columns_df_3D_add] = None

    df_3D = pd.concat([df_3D, df_merge_temp], ignore_index=True)

    for i, row in tqdm(df_3D.iterrows(), total=len(df_3D)):
        img_id = row['img_id']
        sfm_url = row['sfm_url']
        mesh_url = row['mesh_url']
        if sfm_url and mesh_url:
            continue
        img_detail = mapillary_image_client.request_image_details(img_id)
        sfm_cluster = img_detail['sfm_cluster']
        mesh = img_detail['mesh']
        df_sfm_mesh.loc[i, 'sfm_id'] = sfm_cluster['id'] if 'id' in sfm_cluster else None # type: ignore
        df_sfm_mesh.loc[i, 'sfm_url'] = sfm_cluster['url'] if 'url' in sfm_cluster else None # type: ignore
        df_sfm_mesh.loc[i, 'mesh_id'] = mesh['id'] if 'id' in mesh else None # type: ignore
        df_sfm_mesh.loc[i, 'mesh_url'] = mesh['url'] if 'url' in mesh else None # type: ignore
    # df_sfm_mesh.columns
    # df_sfm_mesh.describe()
    # df_sfm_mesh.groupby('sfm_id')['seq_id'].nunique()
    # df_sfm_mesh.groupby('seq_id')['sfm_id'].nunique()
    
    df_3D.to_csv(cfg.path.df_3D, index=False)
    df_3D = df_3D.dropna(subset=['sfm_url', 'mesh_url'], how='any')

    for _, row in tqdm(df_3D.iterrows(), total=len(df_3D)):
        # img_id = row['img_id']
        # seq_id = row['seq_id']
        sfm_id = row['sfm_id']
        sfm_url = row['sfm_url']
        mesh_id = row['mesh_id']
        mesh_url = row['mesh_url']
        fp_sfm = pathlib.Path(os.path.join(cfg.path.dir_sfm, str(sfm_id)))
        if not fp_sfm.exists() and pd.notna(sfm_url):
            mapillary_image_client.download_image_from_url(sfm_url, fp_sfm)
        fp_mesh = pathlib.Path(os.path.join(cfg.path.dir_mesh, str(mesh_id)))
        if not fp_mesh.exists() and pd.notna(mesh_url):
            mapillary_image_client.download_image_from_url(mesh_url, fp_mesh)
    
    return df_3D


def pano_to_subviews(cfg: Config) -> pd.DataFrame:
    scraper = ScrapeImage(cfg)
    df_crossing = prepare_df_crossing(cfg)
    df_image = scraper.load_df_image()
    df_image_panomap = scraper.load_df_image_panomap()
    
    ############### using only actual GPS & highway-xing
    df_crossing = df_crossing[df_crossing['crossingid'].isin(df_image['crossing_id'])]
    # print(df_crossing[df_crossing['STREET'].str.lower().str.contains('wright')])

    ############### using only pano
    df_image = df_image.dropna(subset=['id'])
    df_image['id'] = df_image['id'].astype(int)
    df_image = df_image[df_image['is_pano'] == 1]
    df_image = df_image[df_image['camera_type'] == 'spherical'] # [spherical, equirectangular] equirectangular images require diff view extraction mechanism
    
    ############### merge
    df_image = df_image.merge(df_crossing[['crossingid', 'latitude', 'longitude', 'street']], left_on='crossing_id', right_on='crossingid')
    df_image = df_image.drop(columns=['crossingid'])
    
    ############### using only images with distance over the threshold
    select_xing_within = [0.00000, 0.00001]
    select_img_within = [0.0000, 0.0002]
    # df_image = df_image[(df_image['dist'] <= threshold) | (df_image['computed_dist'] <= threshold)]
    df_min_dist = df_image.loc[df_image.groupby("crossing_id")["dist"].idxmin()][['crossing_id', 'id', 'compass_angle', 'computed_compass_angle', 'sequence', 'lat', 'lon', 'latitude', 'longitude', 'dist']].reset_index(drop=True)
    df_min_dist = df_min_dist[(select_xing_within[0] <= df_min_dist['dist']) & (df_min_dist['dist'] <= select_xing_within[1])]
    # print(df_min_dist)
    
    ############### get image seq
    for i, row in tqdm(df_min_dist.iterrows(), total=df_min_dist.shape[0]):
        crossing_id = row['crossing_id']
        if crossing_id in df_image_panomap['crossing_id'].values:
            continue
        seq_id = row['sequence']
        
        df_panomap_temp = df_image[(df_image['crossing_id'] == crossing_id) & (df_image['sequence'] == seq_id)]
        df_panomap_temp = df_panomap_temp[(df_panomap_temp['computed_compass_angle'].notna())]
        df_panomap_temp = df_panomap_temp[(select_img_within[0] <= df_panomap_temp['dist']) & (df_panomap_temp['dist'] <= select_img_within[1])]
        
        if df_panomap_temp.shape[0] <= 1:
            continue
        
        images = scraper.get_image_seq(seq_id)['data']
        images = [image['id'] for image in images]
        
        df_image_panomap_temp = pd.DataFrame(columns=df_image_panomap.columns)
        for _, seq_row in df_panomap_temp.iterrows():
            img_id = seq_row['id']
            if str(img_id) not in images or pd.isna(seq_row['computed_compass_angle']):
                continue
            img_pos = str(images.index(str(img_id))).zfill(4)
            
            dp_crossing_panomap = os.path.join(cfg.path.dir_image_panomap, seq_id)
            make_dir(dp_crossing_panomap)

            fp_img = os.path.join(cfg.path.dir_scraped_images, f'{img_id}.jpg')
            if not os.path.exists(fp_img):
                continue
            img = np.array(Image.open(fp_img).convert("RGB"))
            for bearing in range(0, 181, cfg.gs.bearing_step):
                view = scraper.extract_view(img, h_fov=cfg.gs.fov, yaw_deg=bearing, pitch_deg=0, out_hw=cfg.gs.panomap_image_hw)
                out_img = Image.fromarray(view)
                fp_img_panomap = os.path.join(dp_crossing_panomap, f'{img_pos}_{img_id}_{bearing:03}.jpg')
                if not os.path.exists(fp_img_panomap):
                    out_img.save(fp_img_panomap)
                df_image_panomap_temp.loc[len(df_image_panomap_temp)] = [crossing_id, seq_id, img_pos, img_id, bearing]
        
        if not df_image_panomap_temp.empty:
            df_image_panomap = pd.concat([df_image_panomap, df_image_panomap_temp])
        
        if i % 10 == 0: # type: ignore
            df_image_panomap.to_csv(cfg.path.df_image_panomap, index=False)
    
    df_image_panomap.to_csv(cfg.path.df_image_panomap, index=False)
    
    ###############
    # img_id = 2726828097607512
    # img_id = 1125481356095921
    # img_id = 375830210537976
    # df_image[df_image['id'] == img_id].iloc[0][['compass_angle', 'computed_compass_angle']]
    # roll, pitch, yaw = df_image[df_image['id'] == img_id].iloc[0]['computed_rotation'] # x(tilt r/l), y(look u/d), z(compass) in 3D (default is east, down, north)
    # roll_deg, pitch_deg, yaw_deg = list(map(math.degrees, [roll, pitch, yaw]))
    # roll_deg, pitch_deg, yaw_deg
    
    return df_image_panomap


if __name__ == '__main__':
    # remove_dir(cfg.path.dir_scraped_images)
    # make_dir(cfg.path.dir_scraped_images)
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('SPRUCE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('CHICAGO')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('IOWA AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('PALM AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('BROCKTON AVE')]
    # df_crossing[df_crossing['STREET'].str.upper().str.contains('WASHINGTON')]
    lat, lon = 33.990282, -117.356688 # SPRUCE ST 0.00002
    lat, lon = 33.997812, -117.348500 # CHICAGO AVE 0.00004
    lat, lon = 33.9942, -117.339849 # IOWA AVE 0.00007
    lat, lon = 33.957269, -117.396787 # BROCKTON AVE 0.00004
    lat, lon = 33.957246, -117.401092 # PALM AVE 0.00003
    lat, lon = 33.938530, -117.396230 # WASHINGTON AVE 0.00003