import os
import time
from transformers import pipeline
import json
import json5
import ast
import shutil
import pandas as pd
from pprint import pprint

def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def remove_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def sanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("/", "--")

def desanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("--", "/")

def generate_openai(client, model_path, content, generation_config=None):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    response = client.responses.create(model=model_path, input=messages)
    output = response.output_text
    return output

def generate_hf(pipe, model_path, content, generation_config={}):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    # response = pipe(messages, return_full_text=False, generate_kwargs=generation_config)
    response = pipe(messages, return_full_text=False, **generation_config)
    output = response[0]['generated_text']
    return output

def generate_google(client, model_path, content, generation_config=None):
    response = client.models.generate_content(
        model=model_path,
        contents=content,
    )
    output = response.text
    return output

def select_generate_func(api):
    if api == 'OpenAI':
        generate_func = generate_openai
    elif api == 'Huggingface':
        generate_func = generate_hf
    elif api == 'Google':
        generate_func = generate_google
    else:
        raise ValueError(f"Unsupported API: {api}")
    return generate_func

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self           # (optional) so you can read .elapsed later
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        elapsed = self.format_hms(self.elapsed)
        print(f"[{self.label}]\t elapsed: {elapsed}")
            
    def format_hms(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60               # still a float now
        return f"{h:02d}:{m:02d}:{s:06.3f}"   # e.g. 00:01:02.357

def as_float(val):
    try:
        return float(val)
    except:
        return val

def as_int(val):
    try:
        return int(val)
    except:
        return val

def lower_str(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val

def parse_json_from_output(output):
    """Parse JSON from the output text of OpenAI response.
    If the output is a plain JSON string, it parses that directly.
    Otherwise, find the code block containing JSON and parse it.
    """
    try:
        if '```' in output:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            str_form57 = output[json_start_index:json_end_index].strip('`')
            if str_form57.startswith('json'):
                str_form57 = str_form57.replace('json', '', 1)
        else:
            str_form57 = output
        try:
            dict_form57 = json.loads(str_form57)
        except:
            dict_form57 = ast.literal_eval(str_form57)
    except:
        dict_form57 = {}
    return dict_form57

def text_binary_classification(pipe, prompt, dict_answer_choice, num_sim):
    list_output = pipe(prompt, max_new_tokens=1, num_return_sequences=num_sim, return_full_text=False)
    list_answer = list(map(lambda output: output['generated_text'].upper(), list_output))
    list_answer_filter = list(filter(lambda answer: answer in dict_answer_choice, list_answer))
    list_answer_map = list(map(lambda answer: dict_answer_choice[answer], list_answer_filter))
    return list_answer_map

def text_generation(pipe, prompt, max_new_tokens=4096):
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    answer = output[0]['generated_text']
    return answer

def prepare_df_record(cfg):
    df_record = pd.read_csv(cfg.path.df_record, parse_dates=['Date'])
    df_record = df_record[df_record['State Name'].str.title().isin(cfg.scrp.target_states)]
    # df_record = df_record[df_record['County Name'].str.title().isin(cfg.scrp.target_counties)]
    df_record = df_record[df_record['Date'] >= cfg.scrp.start_date]
    assert df_record['Report Key'].is_unique
    return df_record

def prepare_df_crossing(cfg):
    df_crossing = pd.read_csv(cfg.path.df_crossing)
    
    # remove API metadata columns
    assert df_crossing[':id'].is_unique
    df_crossing = df_crossing.loc[:, ~df_crossing.columns.astype(str).str.startswith(':')]
    
    # sanity check for duplicated crossingid
    if df_crossing['crossingid'].is_unique is False:
        # leave the latest updated version
        df_dup = df_crossing[df_crossing['crossingid'].duplicated(keep=False)]
        df_dup = df_dup.sort_values(by='revisiondate', ascending=False)
        df_dup_to_remove = df_dup.groupby('crossingid').tail(-1)
        df_crossing = df_crossing.drop(df_dup_to_remove.index)
        assert df_crossing['crossingid'].is_unique
    
    # filter by target states and crossing attributes
    df_crossing = df_crossing[df_crossing['statename'].str.title().isin(cfg.scrp.target_states)]
    # df_crossing = df_crossing[df_crossing['countyname'].str.title().isin(cfg.scrp.target_counties)]
    df_crossing = df_crossing[df_crossing['innear'] == 'In']
    df_crossing = df_crossing[df_crossing['crossingtype'] == 'Public']
    df_crossing = df_crossing[df_crossing['crossingpurpose'] == 'Highway']
    df_crossing = df_crossing[df_crossing['crossingposition'] == 'At Grade']
    df_crossing = df_crossing[df_crossing['crossingclosed'] == 'No']
    df_crossing = df_crossing[~df_crossing['latlongsource'].isin(['Actual'])]
    
    # drop rows with missing latitude, longitude, and geocoded_lat_long
    df_crossing = df_crossing[~df_crossing[['latitude', 'longitude', 'geocoded_lat_long']].isna().all(axis=1)]

    # type cast
    df_crossing[['latitude', 'longitude']] = df_crossing[['latitude', 'longitude']].astype(float)
    
    # # check city description against city, county, state combination
    # df_loc = df_crossing[['latitude', 'longitude', 'geocoded_lat_long', 'statename', 'countyname', 'cityname', 'citydescription']]
    # df_loc_unmatch = df_loc.apply(lambda row: f"{row['cityname']} City, {row['countyname']} County, {row['statename']} State", axis=1) != df_loc['citydescription']
    # print(df_loc[df_loc_unmatch])
    # print(df_loc.loc[df_loc_unmatch & (df_crossing['countyname'] != 'SUTTER'), :])

    # # check geocoded_lat_long against latitude and longitude
    # df_geo = df_crossing[['latitude', 'longitude', 'geocoded_lat_long']]
    # df_geo_unmatch = df_geo['geocoded_lat_long'] != df_geo.apply(lambda row: f'POINT ({row["longitude"]} {row["latitude"]})', axis=1)
    # df_geo[df_geo_unmatch]

    # list_locinfo = ['crossingid', 'statename', 'countyname', 'cityname', 'innear', 'street', 'blocknumber', 'highwayname', 'latitude', 'longitude', 'latlongsource']
    # df_crossing = df_crossing[list_locinfo]
    df_crossing = df_crossing.reset_index(drop=True)

    return df_crossing

def prepare_df_image(cfg):
    df_image = pd.read_csv(cfg.path.df_image)
    df_image = df_image.sort_values('crossing_id', ignore_index=True)
    df_image['computed_rotation'] = df_image['computed_rotation'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image['mesh'] = df_image['mesh'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image['sfm_cluster'] = df_image['sfm_cluster'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    # df_image['detections'] = df_image['detections'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    
    return df_image

def prepare_df_image_seq(cfg):
    df_image_seq = pd.read_csv(cfg.path.df_image_seq)
    return df_image_seq

def prepare_df_image_panomap(cfg):
    df_image_panomap = pd.read_csv(cfg.path.df_image_panomap)
    return df_image_panomap
    
def prepare_df_retrieval(cfg):
    df_retrieval = pd.read_csv(cfg.path.df_retrieval)
    df_retrieval['pub_date'] = pd.to_datetime(df_retrieval['pub_date'])
    assert df_retrieval['news_id'].is_unique, '==========Warning: News is not unique!!!==========='
    return df_retrieval

def prepare_df_match(cfg):
    df_match = pd.read_csv(cfg.path.df_match)
    df_match = df_match[df_match['match'] == 1]
    idx_content_match = df_match.columns.get_loc('content')
    df_match = df_match.iloc[:, :idx_content_match + 1] # type: ignore
    assert df_match['news_id'].is_unique, '==========Warning: News is not unique!!!==========='
    return df_match

def prepare_dict_col_indexing(cfg):
    with open(cfg.path.dict_col_indexing, 'r') as f:
        dict_col_indexing = json5.load(f)
    return dict_col_indexing

def prepare_dict_idx_mapping(cfg):
    with open(cfg.path.dict_idx_mapping, 'r') as f:
        dict_idx_mapping = json5.load(f)
        dict_idx_mapping_inverse = {v: k for k, v in dict_idx_mapping.items()}
        if '' in dict_idx_mapping_inverse:
            dict_idx_mapping_inverse.pop('')
    return dict_idx_mapping, dict_idx_mapping_inverse

def prepare_dict_answer_places(cfg):
    with open(cfg.path.dict_answer_places, 'r') as f:
        dict_answer_places = json5.load(f)
    return dict_answer_places

def prepare_dict_form57(cfg):
    with open(cfg.path.form57_json, 'r') as f:
        dict_form57 = json.load(f)
    return dict_form57

def prepare_dict_form57_group(cfg):
    with open(cfg.path.form57_json_group, 'r') as f:
        dict_form57_group = json5.load(f)
    return dict_form57_group


def __EDA_df_crossing(cfg):
    df_crossing_raw = prepare_df_crossing(cfg)
    list_loc_info = ['innear', 'statename', 'countyname', 'cityname', 'citydescription', 'street', 'highwayname', 'latitude', 'longitude', 'latlongsource']
    list_crossing_attr = ['crossingtype', 'developmenttype', 'highwaypaved']
    list_train_attr = ['typeoftrainserviceids']
    list_all_attr = list_loc_info + list_crossing_attr + list_train_attr
    df_crossing = df_crossing_raw[list_all_attr]

    ############### attributes(= column names) explored

    ### innear: 'In' indicates the crossing is within a city, while 'Near' indicates it is near a city, possibly in census-designated places(CDP) or unincorporated areas.
    df_crossing['innear'].value_counts()
    print(df_crossing[list_loc_info])

    ### cityname: The name of the city/CDP where the crossing is located (if innear is 'In') or the nearest city (if innear is 'Near').
    df_crossing_raw[['cityname', 'citycode']]

    ### citydescription: Follows the format "City/CDP, County, State". However, about 1% of the entries mismatch the county information because countyname 
    df_crossing[df_crossing['citydescription'].str.split(', ', expand=True)[0].str.replace(' City', '', regex=False) != df_crossing['cityname']]
    print(df_crossing_raw[df_crossing['citydescription'].str.split(', ', expand=True)[1].str.replace(' County', '', regex=False) == df_crossing['countyname']][list_loc_info])
    df_crossing[df_crossing['citydescription'].str.split(', ', expand=True)[2].str.replace(' State', '', regex=False) != df_crossing['statename']]
    
    ### street: If the street name contains 'private', over 99% of them are with 'Private' crossing type. So, crossing type is a strong indicator of private crossings.
    is_private_street = df_crossing['street'].str.contains('private', case=False)
    is_private_crossing = df_crossing['crossingtype'].eq('Private')
    is_private_street.sum()
    is_private_crossing.sum()
    df_crossing[is_private_street | is_private_crossing]
    print(df_crossing[is_private_street & ~is_private_crossing])
    
    ### highwayname: 
    ### crossingtype: Indicates whether the crossing is 'Public' or 'Private'. 'Private' crossings are typically 

    ### latlongsource: If 'NaN', they are very unreliable.
    print(df_crossing[df_crossing['latlongsource'].isin([pd.NA])][list_loc_info].sample(15))
                      
    # highwaypaved: Related to crossingtype. If crossingtype is 'Public', highwaypaved is most likely to be 'Yes' while 'Private' crossings are not necessarily paved.
    print(df_crossing.sample(15)[['street', 'highwayname', 'latitude', 'longitude', 'latlongsource', 'crossingtype', 'typeoftrainserviceids', 'developmenttype']])
    print(df_crossing[df_crossing['latlongsource'].isin([pd.NA])][['latitude', 'longitude', 'crossingtype', 'typeoftrainserviceids', 'developmenttype']].sample(10))
    print(df_crossing.loc[184965, list_loc_info + ['movementsperday']])
    print(df_crossing.loc[184965:184965, list_loc_info + ['typeoftrainserviceids', 'ltpassengertrainperday', 'numberpassengertrainperday']])
