"""Selectively acquire the pinned 697-frame Eigen split from public KITTI raw ZIPs."""
import argparse
from collections import OrderedDict,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile
import numpy as np
from PIL import Image
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project import write_json
from src.model import sha256
BASE='https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
SPLIT_SHA='e226d4ca16e0e6e9695435014347dcf973588531330536fd8a68d298ef091a6b'
UTILS_SHA='7f927265725f3364542c94cd643c8b7f0350ddc0abaf5dc08206504185827835'

class RangeReader(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.session=requests.Session();self.position=0;self.cache=OrderedDict();self.bytes_received=0
        response=self.session.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},stream=True,timeout=(15,60))
        try:
            if response.status_code!=206:raise RuntimeError('Server must support bounded byte ranges: '+str(response.status_code))
            match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
            if not match:raise RuntimeError('Invalid ZIP range size')
            self.size=int(match.group(1));self.etag=response.headers.get('ETag')
            if not self.etag:raise RuntimeError('Archive identity ETag required')
        finally:response.close()
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0:raise ValueError('Negative ZIP seek')
        self.position=position;return position
    def block(self,index):
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        start=index*2**21;end=min(self.size,start+2**21)-1
        for attempt in range(3):
            response=None
            try:
                response=self.session.get(self.url,headers={'Range':'bytes={}-{}'.format(start,end),'If-Match':self.etag,'Accept-Encoding':'identity'},stream=True,timeout=(15,60))
                if response.status_code!=206 or response.headers.get('Content-Range')!='bytes {}-{}/{}'.format(start,end,self.size) or response.headers.get('ETag')!=self.etag:raise RuntimeError('ZIP range/identity mismatch')
                data=response.raw.read(end-start+2,decode_content=False)
                if len(data)!=end-start+1:raise RuntimeError('Truncated ZIP range')
                self.bytes_received+=len(data);self.cache[index]=data
                if len(self.cache)>8:self.cache.popitem(last=False)
                return data
            except (requests.RequestException,RuntimeError):
                if attempt==2:raise
                time.sleep(attempt+1)
            finally:
                if response is not None:response.close()
    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        if size>16*2**20:raise ValueError('Refusing an unbounded ZIP read')
        chunks=[]
        while size>0:
            index=self.position//2**21;offset=self.position%2**21;data=self.block(index);chunk=data[offset:offset+size]
            if not chunk:break
            chunks.append(chunk);self.position+=len(chunk);size-=len(chunk)
        return b''.join(chunks)
    def close(self):self.session.close();self.cache.clear();super().close()


def reference_generator():
    source=ROOT/'third_party/monodepth2/kitti_utils.py'
    if sha256(source)!=UTILS_SHA:raise ValueError('Pinned KITTI projection source changed')
    text=source.read_text(encoding='utf-8').replace('np.int)','int)')
    path=ROOT/'.cache/eigen_kitti_utils_compat.py';path.parent.mkdir(exist_ok=True);path.write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('eigen_kitti_utils_compat',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.generate_depth_map,sha256(path)


def extract(reader,names,root):
    records={}
    with zipfile.ZipFile(reader) as archive:
        for name in names:
            info=archive.getinfo(name)
            if info.file_size>16*2**20:raise ValueError('Unexpected oversized selected member')
            target=(root/name).resolve()
            if root.resolve() not in target.parents:raise ValueError('Unsafe member path')
            data=archive.read(name) # ZipFile verifies selected-member CRC.
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);records[name]=sha256(target)
    return records


def prepare(output,workers=3,resume=False):
    output=Path(output).resolve();staged=output.with_name(output.name+'.incomplete')
    if output.exists() or (staged.exists() and not resume):raise FileExistsError('Fresh output required; --resume permits incomplete acquisition only')
    split=ROOT/'third_party/monodepth2/splits/eigen/test_files.txt'
    if sha256(split)!=SPLIT_SHA:raise ValueError('Pinned Eigen split changed')
    lines=split.read_text().splitlines()
    if len(lines)!=697 or len(set(lines))!=697:raise ValueError('Need the complete 697-row split')
    groups=defaultdict(list)
    for line in lines:
        drive,frame,side=line.split()
        if not re.fullmatch(r'\d{4}_\d{2}_\d{2}/\d{4}_\d{2}_\d{2}_drive_\d{4}_sync',drive) or side!='l':raise ValueError('Unexpected Eigen split identity')
        groups[drive].append(int(frame))
    staged.mkdir(parents=True,exist_ok=resume);(staged/'indexes').mkdir(exist_ok=True)
    generate,patched_sha=reference_generator();archives=[]
    for date in sorted({drive.split('/')[0] for drive in groups}):
        with RangeReader(BASE+date+'_calib.zip') as reader:
            records=extract(reader,[date+'/'+name for name in ['calib_cam_to_cam.txt','calib_velo_to_cam.txt','calib_imu_to_velo.txt']],staged)
            archives.append(dict(url=reader.url,etag=reader.etag,archive_bytes=reader.size,received_bytes=reader.bytes_received,selected_files=records))
    def drive_job(item):
        drive,frames=item;date,short=drive.split('/');index=staged/'indexes'/(short+'.json')
        if index.exists():
            record=json.loads(index.read_text(encoding='utf-8'))
            if all(sha256(staged/p)==digest for p,digest in record['files'].items()):return record
            raise ValueError('Incomplete-cache checksum mismatch')
        stem=short[:-5];url=BASE+stem+'/'+short+'.zip'
        names=[drive+'/'+kind+'/'+str(frame).zfill(10)+suffix for frame in sorted(frames) for kind,suffix in [('image_02/data','.png'),('velodyne_points/data','.bin')]]
        with RangeReader(url) as reader:
            files=extract(reader,names,staged);archive=dict(url=url,etag=reader.etag,archive_bytes=reader.size,received_bytes=reader.bytes_received)
        samples=[]
        for frame in sorted(frames):
            rgb=drive+'/image_02/data/'+str(frame).zfill(10)+'.png';lidar=drive+'/velodyne_points/data/'+str(frame).zfill(10)+'.bin'
            points=np.fromfile(staged/lidar,dtype=np.float32)
            if points.size%4 or not np.isfinite(points).all():raise ValueError('Invalid selected LiDAR')
            depth=generate(str(staged/date),str(staged/lidar),2,True).astype(np.float32)
            with Image.open(staged/rgb) as image:
                if image.size!=(depth.shape[1],depth.shape[0]):raise ValueError('Unregistered Eigen RGB/reference dimensions')
            relative='depth/'+short+'_'+str(frame).zfill(10)+'.npy';target=staged/relative;target.parent.mkdir(exist_ok=True);np.save(target,depth,allow_pickle=False);files[relative]=sha256(target)
            samples.append(dict(id=short+'_'+str(frame).zfill(10),scene=short,frame=frame,rgb=rgb,rgb_sha256=files[rgb],depth=relative,depth_sha256=files[relative],lidar=lidar,lidar_sha256=files[lidar]))
        result=dict(drive=drive,files=files,samples=samples,archive=archive);write_json(index,result);print('EIGEN DRIVE',short,len(samples),flush=True);return result
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(drive_job,item) for item in groups.items()]
        for future in as_completed(futures):records.append(future.result())
    by_id={s['scene']+'_'+str(s['frame']).zfill(10):s for record in records for s in record['samples']}
    samples=[by_id[drive.split('/')[1]+'_'+str(int(frame)).zfill(10)] for drive,frame,side in [line.split() for line in lines]]
    manifest=dict(schema_version=1,dataset='KITTI Eigen 697 pinned split',split_sha256=SPLIT_SHA,source_revision='b017a8164fae339b0ce01f3a9e5048104d5abd0e',source_utils_sha256=UTILS_SHA,compat_utils_sha256=patched_sha,compat_patch='Replace removed np.int aliases with builtin int; numerical projection unchanged',reference='Pinned export_gt_depth.py convention: generate_depth_map(camera=2,vel_depth=True); collision minimum, native dimensions',protocol=dict(min_depth=0.001,max_depth=80,crop='garg',scaling='per_image_median',input_resize='PIL LANCZOS to 640x192; checkpoint-internal normalization',post_processing=False),source='KITTI public raw archive endpoints used by upstream download scripts; retain CC BY-NC-SA attribution/terms',samples=samples,archives=archives+[x['archive'] for x in records],limits='Fixed test split; prior 24-image development overlap must be audited before interpreting independence. No raw/reference data is used for training.')
    if len(samples)!=697:raise ValueError('Incomplete Eigen data cannot be published')
    write_json(staged/'evaluation_manifest.json',manifest);staged.replace(output);print('EIGEN DATA COMPLETE',len(samples),flush=True);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');a=p.parse_args();prepare(a.output,a.workers,a.resume)
