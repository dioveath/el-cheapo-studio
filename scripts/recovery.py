import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

BASE = os.getenv('STUDIO_URL', 'http://localhost:3000')
ROOT = Path(__file__).resolve().parent.parent
report = json.loads((ROOT / 'data/verification.json').read_text())
source_id = report['batch'][0]['id']
results = {}

def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)

def compose(*args):
    subprocess.run(['docker', 'compose', *args], cwd=ROOT, check=True)

def wait(id, condition, timeout=240):
    end = time.monotonic()+timeout
    last = None
    while time.monotonic()<end:
        try:
            job = request('/api/jobs/'+id)
        except Exception:
            time.sleep(1)
            continue
        state = (job['status'],job['phase'],job['attempt'])
        if state!=last:
            print(id,state,flush=True)
            last = state
        if condition(job):
            return job
        if job['status'] in ('failed','cancelled'):
            raise AssertionError(job)
        time.sleep(.4)
    raise AssertionError('Job timed out')


(ROOT / 'artifacts').mkdir(exist_ok=True)
compose('stop','worker')
variant = request(f'/api/jobs/{source_id}/variants', {'target_vertices':250,'faceted':False})['jobs'][0]
compose('restart','server')
compose('start','worker')
result = wait(variant['id'],lambda j:j['status']=='completed')
assert result['stats']['reconstruction_reused'] and result['stats']['reconstruction_seconds']==0
results['restart_variant']={'id':result['id'],'stats':result['stats']}
source=request('/api/jobs/'+source_id)

job=request('/api/jobs',{'asset_ids':[source['asset_id']],'settings':{'seed':987,'steps':10,'resolution':128}})['jobs'][0]
wait(job['id'],lambda j:'step' in j['phase'])
compose('kill','-s','SIGKILL','worker')
compose('start','worker')
result=wait(job['id'],lambda j:j['status']=='completed')
assert result['attempt']>=2, result
results['worker_crash_recovery']={'id':result['id'],'attempt':result['attempt'],'stats':result['stats']}

job=request('/api/jobs',{'asset_ids':[source['asset_id']],'settings':{'seed':988}})['jobs'][0]
wait(job['id'],lambda j:'step' in j['phase'])
request('/api/jobs/'+job['id']+'/cancel',{})
result=wait(job['id'],lambda j:j['status']=='cancelled')
assert result['preview_url'] is None
results['active_cancellation']={'id':result['id'],'status':result['status']}
(ROOT/'artifacts/recovery.json').write_text(json.dumps(results,indent=2))
print('PASS: API restart, persistent original reuse, hard worker crash retry, active cancellation.',flush=True)
