"""No-network range/CRC and pinned-projection checks; not KITTI score evidence."""
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from tools import prepare_kitti_eigen as importer
from project import ROOT

class Raw(io.BytesIO):
    def read(self,size=-1,decode_content=False):return super().read(size)
class Response:
    def __init__(self,data,start,end,etag='fixture'):
        self.status_code=206;self.headers={'Content-Range':'bytes {}-{}/{}'.format(start,end,len(data)),'ETag':etag};self.raw=Raw(data[start:end+1])
    def close(self):pass
class Session:
    def __init__(self,data,bad_identity=False):self.data=data;self.calls=0;self.bad_identity=bad_identity
    def get(self,url,headers,**kwargs):
        a,b=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',headers['Range']).groups());self.calls+=1
        return Response(self.data,a,b,'changed' if self.bad_identity and self.calls>1 else 'fixture')
    def close(self):pass
class EigenAcquisitionTests(unittest.TestCase):
    def test_bounded_seekable_zip_reads_and_crc_extraction(self):
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('date/example.txt',b'example contents')
        session=Session(memory.getvalue())
        with patch.object(importer.requests,'Session',return_value=session):
            with importer.RangeReader('https://fixture.invalid/archive.zip') as stream:
                with zipfile.ZipFile(stream) as z:self.assertEqual(z.read('date/example.txt'),b'example contents')
                self.assertLessEqual(stream.bytes_received,len(memory.getvalue()))
                stream.seek(0);self.assertEqual(stream.read(2),b'PK')
    def test_archive_identity_changes_fail_without_bulk_fallback(self):
        with patch.object(importer.requests,'Session',return_value=Session(b'abcdefgh',True)),patch.object(importer.time,'sleep'):
            with importer.RangeReader('https://fixture.invalid/archive.zip') as stream:
                with self.assertRaisesRegex(RuntimeError,'identity'):stream.read(3)
    def test_official_projection_compatibility_keeps_duplicate_minimum(self):
        generate,_=importer.reference_generator()
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tmp')) as temporary:
            root=Path(temporary)
            (root/'calib_cam_to_cam.txt').write_text('S_rect_02: 8 6\nR_rect_00: 1 0 0 0 1 0 0 0 1\nP_rect_02: 1 0 0 0 0 1 2 0 0 0 1 0\n')
            (root/'calib_velo_to_cam.txt').write_text('R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n')
            points=np.array([[2,0,1,.5],[1,0,.5,.7],[-1,0,1,.3]],dtype=np.float32);points.tofile(root/'points.bin')
            depth=generate(str(root),str(root/'points.bin'),2,True)
            self.assertEqual(depth.shape,(6,8));self.assertEqual(depth[1,1],1);self.assertEqual(np.count_nonzero(depth),1)
