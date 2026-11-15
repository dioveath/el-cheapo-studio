import io
import json
from pathlib import Path
import time
import httpx
from PIL import Image
c=httpx.Client(base_url='http://web:3000',timeout=90)
def post(path,**kw):
    r=c.post(path,**kw);r.raise_for_status();return r.json()
def wait(id):
    until=time.monotonic()+240
    last=None
    while time.monotonic()<until:
        j=c.get('/api/jobs/'+id).json()
        state=(j['status'],j['phase'])
        if state!=last:print(state,flush=True);last=state
        if j['status'] in ('completed','failed','cancelled'):
            assert j['status']=='completed',j
            return j
        time.sleep(1)
    raise AssertionError('Timeout')
source=json.loads(Path('/data/verification.json').read_text())['batch'][0]
mesh=c.get('/api/jobs/'+source['id']+'/files/glb').content
asset=post('/api/assets',files={'file':('imported-character.glb',mesh)})
job=post('/api/jobs',json={'asset_ids':[asset['id']], 'settings':{'target_vertices':250}})['jobs'][0]
result=wait(job['id'])
assert result['stats']['model']=='Imported mesh'
assert result['stats']['original_vertices']==1000
assert result['stats']['watertight']
image=Image.open('/app/examples/character.png').convert('RGBA')
background=Image.new('RGBA',image.size,'white');background.alpha_composite(image)
buffer=io.BytesIO();background.convert('RGB').save(buffer,format='PNG')
asset=post('/api/assets',files={'file':('opaque-character.png',buffer.getvalue(),'image/png')})
job=post('/api/jobs',json={'asset_ids':[asset['id']],'settings':{'steps':10,'resolution':128}})['jobs'][0]
opaque=wait(job['id'])
assert opaque['stats']['cpu_offload'] is False
Path('/data/import-verification.json').write_text(json.dumps({'import':result,'opaque_image':opaque},indent=2))
print('PASS: faceted GLB import recovers shared vertices; opaque image background removal and GPU generation.',flush=True)
