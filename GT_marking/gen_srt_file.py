import datetime

import srt
from datetime import timedelta, datetime
import pandas as pd
import glob


annot_files = glob.glob('../cache/real_world_dataset/p5/exported/*.txt')
for annot_file in annot_files:
    srt_file = annot_file[:-4]+'.srt'
    df_annot = pd.read_csv(annot_file, sep='\t',
                           names=['type', 'unamed', 'start_time', 'end_time', 'duration', 'label'])

    subtitles = []

    for row_idx, row in df_annot.iterrows():
        t_row_start = datetime.strptime(row['start_time'], '%H:%M:%S.%f')
        t_row_end = datetime.strptime(row['end_time'], '%H:%M:%S.%f')
        t_row_start_delta = timedelta(hours=t_row_start.hour, minutes=t_row_start.minute, seconds=t_row_start.second)
        t_row_end_delta = timedelta(hours=t_row_end.hour, minutes=t_row_end.minute, seconds=t_row_end.second)
        position_val = '{\\an9}'
        if row['type']=='context':
            position_val = '{\\an3}'
        row_sub = srt.Subtitle(index=row_idx,
                               start=t_row_start_delta,
                               end=t_row_end_delta,
                               content=position_val + row['type'] + ":" + row['label'])
        subtitles.append(row_sub)

    composed_subtitles = srt.compose(subtitles)
    with open(srt_file,"w") as f:
        f.write(composed_subtitles)
    print(f"Done annotation file {annot_file}...")
print('finished')
