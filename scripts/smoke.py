import json
from pathlib import Path
import time
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60)
report = {}

def post(path, **kwargs):
    response = client.post(path, **kwargs)
    response.raise_for_status()
    return response.json()

def wait(ids, timeout=240):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        jobs = [client.get(f"/api/jobs/{id}").json() for id in ids]
        phases = [(j['status'], j['phase']) for j in jobs]
        if phases != last:
            print(phases, flush=True)
            last = phases
        assert sum(j['status'] == 'running' for j in jobs) <= 1, "More than one GPU job running"
        if all(j['status'] in {'completed','failed','cancelled'} for j in jobs):
            return jobs
        time.sleep(1)
    raise AssertionError("Jobs timed out")

assert client.get('/api/health').status_code == 200
assert client.post('/api/assets',files={'file':('bad.png',b'not an image')}).status_code == 400
assert client.post('/api/assets',files={'file':('bad.txt',b'invalid')}).status_code == 415
assert client.post('/api/jobs',json={'asset_ids':[]}).status_code == 422
assets = []
for name in ['character','chair']:
    assets.append(post('/api/assets',files={'file':(name+'.png',Path('/app/examples/'+name+'.png').read_bytes(),'image/png')})['id'])
batch = post('/api/jobs',json={'asset_ids':assets})
ids = [j['id'] for j in batch['jobs']]
print('BATCH',ids,flush=True)
jobs = wait(ids)
assert all(j['status']=='completed' for j in jobs), jobs
report['batch'] = [{'id': j['id'], 'name': j['filename'], **j['stats']} for j in jobs]
for job in jobs:
    assert job['stats']['cpu_offload'] is False
    assert job['stats']['vertices'] > 0
    for kind in ['glb','obj','original']:
        response = client.get(f"/api/jobs/{job['id']}/files/{kind}")
        assert response.status_code == 200
        assert response.content.startswith(b'glTF') if kind != 'obj' else b'\nv ' in response.content
variant = post(f'/api/jobs/{ids[0]}/variants',json={'target_vertices':500,'faceted':True})['jobs'][0]
result = wait([variant['id']])[0]
assert result['status'] == 'completed', result
assert result['stats']['reconstruction_reused'] is True
assert result['stats']['reconstruction_seconds'] == 0
assert result['reconstruction_key'] == jobs[0]['reconstruction_key']
report['variant'] = {'id':result['id'], **result['stats']}

invalid = post('/api/assets',files={'file':('invalid.obj', b'v 0 0 0\n')})
failed = post('/api/jobs',json={'asset_ids':[invalid['id']]})['jobs'][0]
failed = wait([failed['id']])[0]
assert failed['status'] == 'failed' and failed['error'], failed
retried = post(f"/api/jobs/{failed['id']}/retry")['jobs'][0]
assert retried['id'] != failed['id']
assert wait([retried['id']])[0]['status'] == 'failed'

cancelled = post('/api/jobs',json={'asset_ids':assets, 'settings':{'seed':43}})['jobs']
for job in cancelled:
    post(f"/api/jobs/{job['id']}/cancel")
assert all(j['status']=='cancelled' for j in wait([j['id'] for j in cancelled]))
report['checks'] = ['batch','single GPU activity','GLB/OBJ/original downloads','persistent shape variant','upload validation','failure','retry','queued cancellation']
Path('/data/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
