from urllib.parse import urlencode
import requests
import py360convert
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import subprocess
import platform
import copy
import pathlib
from .config import Config
from .utils import (as_int, as_float, prepare_df_image_panomap, remove_dir, make_dir,
                    prepare_df_record, prepare_df_crossing, prepare_df_image, prepare_df_image_seq)
from pprint import pprint
from scipy.spatial.distance import cdist
import io
from PIL import Image
import math


class ScrapeImage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = self.cfg.apikey.mapillary
        self.img_search_fields = ','.join(self.cfg.scrp.img_search_fields)
        self.img_detail_fields = ','.join(self.cfg.scrp.img_detail_fields)
        self.df_image = None
    
    def load_df_image(self):
        if os.path.exists(self.cfg.path.df_image):
            df_image = prepare_df_image(self.cfg)
        else:
            cols = list(self.cfg.scrp.img_detail_fields)
            cols.remove('geometry')
            cols.remove('computed_geometry')
            cols.extend(['lon', 'lat', 'computed_lon', 'computed_lat', 'dist', 'computed_dist'])
            cols = ['crossing_id'] + cols
            df_image = pd.DataFrame(columns=cols)
            df_image.to_csv(self.cfg.path.df_image, index=False)
        return df_image
    
    def load_df_image_seq(self):
        if os.path.exists(self.cfg.path.df_image_seq):
            df_image_seq = prepare_df_image_seq(self.cfg)
        else:
            cols = ['crossing_id', 'seq_id', 'img_pos', 'img_id', 'bearing']
            df_image_seq = pd.DataFrame(columns=cols)
            df_image_seq.to_csv(self.cfg.path.df_image_seq, index=False)
        return df_image_seq

    def load_df_image_panomap(self):
            if os.path.exists(self.cfg.path.df_image_panomap):
                df_image_panomap = prepare_df_image_panomap(self.cfg)
            else:
                cols = ['crossing_id', 'seq_id', 'img_pos', 'img_id', 'bearing']
                df_image_panomap = pd.DataFrame(columns=cols)
                df_image_panomap.to_csv(self.cfg.path.df_image_panomap, index=False)
            return df_image_panomap
        
    def search_images(self, bbox: str, limit: int = 10):
        """
        Query Mapillary for images inside a bounding box.
        Returns a list of image objects with basic metadata.
        """
        url = (
            "https://graph.mapillary.com/images"
            f"?access_token={self.api_key}"
            f"&bbox={bbox}"
            f"&fields={self.img_search_fields}"
            f"&limit={limit}"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Mapillary returns {"data": [ ...images... ], ...maybe paging...}
        return data.get("data", [])

    def get_image_details(self, image_id: str):
        """
        Ask Mapillary for richer metadata for one specific image.
        Returns a dict with fields we asked for, including thumb_1024_url.
        """
        url = (
            f"https://graph.mapillary.com/{image_id}"
            f"?access_token={self.api_key}"
            f"&fields={self.img_detail_fields}"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def download_thumbnail(self, image_url: str, out_path: pathlib.Path):
        """
        Download the thumbnail JPG and save it locally.
        """
        resp = requests.get(image_url)
        resp.raise_for_status()

        out_path.write_bytes(resp.content)
        return out_path

    def get_image_seq(self, seq_id: str):
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

def scrape_image(cfg: Config) -> pd.DataFrame:
    df_crossing = prepare_df_crossing(cfg)
    scraper = ScrapeImage(cfg)
    df_image = scraper.load_df_image()
    for i, row in tqdm(df_crossing[['crossingid', 'latitude', 'longitude']].iterrows(), total=df_crossing.shape[0]):
        crossing, lat, lon = row
        if crossing in df_image['crossing_id'].values:
            continue
        bbox_exact_match = f"{lon - cfg.scrp.bbox_offset},{lat - cfg.scrp.bbox_offset},{lon + cfg.scrp.bbox_offset},{lat + cfg.scrp.bbox_offset}"
        imgs = scraper.search_images(bbox_exact_match, limit=cfg.scrp.n_img)

        details_concat = [{'crossing_id': crossing}]
        for img in imgs:
            img_id = img["id"]
            if img_id in df_image['id'].values:
                continue
            details = scraper.get_image_details(img_id)
            if details.get('geometry', None):
                assert details['geometry']['type'] == 'Point'
                geometry = details.pop('geometry')
                details['lon'] = geometry['coordinates'][0]
                details['lat'] = geometry['coordinates'][1]
                dist = ((lat - details['lat'])**2 + (lon - details['lon'])**2)**0.5
                details['dist'] = dist
                assert dist <= cfg.scrp.bbox_offset * 2**0.5
                if dist > cfg.scrp.bbox_offset:
                    continue
            if details.get('computed_geometry', None):
                assert details['computed_geometry']['type'] == 'Point'
                computed_geometry = details.pop('computed_geometry')
                details['computed_lon'] = computed_geometry['coordinates'][0]
                details['computed_lat'] = computed_geometry['coordinates'][1]
                computed_dist = ((lon - details['computed_lon'])**2 + (lat - details['computed_lat'])**2)**0.5
                details['computed_dist'] = computed_dist
            if details.get('computed_rotation', None):
                assert isinstance(details['computed_rotation'], list)
            # print(f"computed:\t{computed_dist :.6f}")
            # print(f"actual:\t{dist :.6f}")
            # pprint(details, sort_dicts=False)

            details['crossing_id'] = crossing
            details_concat.append(details)
        
        df_image_temp = pd.DataFrame(details_concat, columns=df_image.columns)
        if not df_image_temp.empty:
            df_image = pd.concat([df_image, df_image_temp], ignore_index=True)
        
        if i % 10 == 0: # type: ignore
            df_image.to_csv(cfg.path.df_image, index=False)
        
    df_image.to_csv(cfg.path.df_image, index=False)

    for i, row in tqdm(df_image.iterrows(), total=df_image.shape[0]):
        img_id = as_int(row['id'])
        thumb_url = row["thumb_original_url"]
        if pd.isna(img_id):
            continue
        output_file = pathlib.Path(os.path.join(cfg.path.dir_scraped_images, f"{img_id}.jpg"))
        if not output_file.exists() and pd.notna(thumb_url):
            scraper.download_thumbnail(thumb_url, output_file)

    return df_image

def scrape_image_seq(cfg: Config) -> pd.DataFrame:
    scraper = ScrapeImage(cfg)
    df_crossing = prepare_df_crossing(cfg)
    df_image = scraper.load_df_image()
    df_image_seq = scraper.load_df_image_seq()
    
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
    select_img_within = [0.0001, 0.0002]
    # df_image = df_image[(df_image['dist'] <= threshold) | (df_image['computed_dist'] <= threshold)]
    df_min_dist = df_image.loc[df_image.groupby("crossing_id")["dist"].idxmin()][['crossing_id', 'id', 'compass_angle', 'computed_compass_angle', 'sequence', 'lat', 'lon', 'latitude', 'longitude', 'dist']].reset_index(drop=True)
    df_min_dist = df_min_dist[(select_xing_within[0] <= df_min_dist['dist']) & (df_min_dist['dist'] <= select_xing_within[1])]
    # print(df_min_dist)
    
    ############### get image seq
    (((df_image['lat'] - df_image['computed_lat'])**2 + (df_image['lon'] - df_image['computed_lon'])**2)**0.5).dropna().sort_values()
    for i, row in tqdm(df_min_dist.iterrows(), total=df_min_dist.shape[0]):
        crossing_id = row['crossing_id']
        if crossing_id in df_image_seq['crossing_id'].values:
            continue
        seq_id = row['sequence']
        xing_lat, xing_lon = row[['latitude', 'longitude']]
        
        df_seq_temp = df_image[(df_image['crossing_id'] == crossing_id) & (df_image['sequence'] == seq_id)]
        df_seq_temp = df_seq_temp[(df_seq_temp['computed_compass_angle'].notna())]
        df_seq_temp = df_seq_temp[(select_img_within[0] <= df_seq_temp['dist']) & (df_seq_temp['dist'] <= select_img_within[1])]
        
        if df_seq_temp.shape[0] <= 1:
            continue
        
        images = scraper.get_image_seq(seq_id)['data']
        images = [image['id'] for image in images]
        # for image in tqdm(images, leave=False):
        #     img_id = image['id']
        #     details = scraper.get_image_details(img_id)
        #     dist = ((row[['LONGITUD', 'LATITUDE']] - details['geometry']['coordinates'])**2).sum()**0.5
        #     if dist > cfg.scrp.bbox_offset * 2 or dist < cfg.scrp.bbox_offset / 2:
        #         continue
        #     assert details['geometry']['type'] == 'Point'
        #     geometry = details.pop('geometry')
        #     details['lon'] = geometry['coordinates'][0]
        #     details['lat'] = geometry['coordinates'][1]
        
        # print(df_seq_temp)
        df_image_seq_temp = pd.DataFrame(columns=df_image_seq.columns)
        for _, seq_row in df_seq_temp.iterrows():
            img_id = seq_row['id']
            if str(img_id) not in images or pd.isna(seq_row['computed_compass_angle']):
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
            view = scraper.extract_view(img, h_fov=90, yaw_deg=bearing, pitch_deg=0, out_hw=(720, 960))
            out_img = Image.fromarray(view)
            dp_crossing_seq = os.path.join(cfg.path.dir_image_seq, seq_id)
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
    
    ###############
    # img_id = 2726828097607512
    # img_id = 1125481356095921
    # img_id = 375830210537976
    # df_image[df_image['id'] == img_id].iloc[0][['compass_angle', 'computed_compass_angle']]
    # roll, pitch, yaw = df_image[df_image['id'] == img_id].iloc[0]['computed_rotation'] # x(tilt r/l), y(look u/d), z(compass) in 3D (default is east, down, north)
    # roll_deg, pitch_deg, yaw_deg = list(map(math.degrees, [roll, pitch, yaw]))
    # roll_deg, pitch_deg, yaw_deg
    
    return df_image_seq


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