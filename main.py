import os
from modules import (
    build_config,
    scrape_image, scrape_image_seq, pano_to_subviews,
    run_COLMAP, visualize_COLMAP, train_gs, visualize_gs
)

print('###########################################################################')
print('###########################################################################')


############### config
cfg = build_config()
print('------------Configuration DONE!!------------')


############### scrape crossing images from mapillary (ONLY ONE-TIME TASK FOR EVALUATION)
# df_image = scrape_image(cfg)
# df_image_seq = scrape_image_seq(cfg)
# df_image_panomap = pano_to_subviews(cfg)
print('------------Scraping Images DONE!!------------')


############### 3D reconstruction
target_seq_id = 'Sw3JZqKE9vmVxRfA0OPbnr'
# run_COLMAP(cfg, target_seq_id)
# visualize_COLMAP(cfg, target_seq_id)
train_gs(cfg, target_seq_id)
visualize_gs(cfg, target_seq_id)

print('###########################################################################')
print('###########################################################################')
