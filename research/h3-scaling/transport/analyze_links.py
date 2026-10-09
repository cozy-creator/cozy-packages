"""Summarize NVLink counters with complete inventory and explicit sampling windows."""
from pathlib import Path
import argparse,json,re,statistics


def counters(text):
    gpu=None; result={}
    for line in text.splitlines():
        m=re.match(r'GPU (\d+):.*UUID: ([^)]+)',line)
        if m: gpu=m[2]
        m=re.search(r'Link (\d+): (Data|Raw) (Tx|Rx): (\d+) KiB',line)
        if m and gpu: result[(gpu,int(m[1]),m[3])]=int(m[4])
    return result


def inventory(receipt, assigned):
    if set(receipt['gpu_uuids']) != set(assigned):
        raise ValueError('Sampler inventory does not match the assigned GPU UUIDs')
    active=set();gpu=None
    for line in receipt['counter_feasibility']['status']['stdout'].splitlines():
        m=re.match(r'GPU (\d+):.*UUID: ([^)]+)',line)
        if m: gpu=m[2]
        m=re.search(r'Link (\d+): [\d.]+ GB/s',line)
        if m and gpu:active.add((gpu,int(m[1])))
    if {g for g,_ in active} != set(assigned):
        raise ValueError('Active-link status does not cover every assigned GPU')
    expected={(g,link,direction) for g,link in active for direction in ('Tx','Rx')}
    result={}
    for kind,name in [('d','data'),('r','raw')]:
        probe=receipt['counter_feasibility'][name]
        result[kind]=expected if probe['exit_code']==0 and set(counters(probe['stdout']))==expected else None
    return result


def stats(values):
    return {'min':min(values),'mean':statistics.mean(values),'max':max(values)} if values else None


def delta(samples, expected):
    if expected is None:raise ValueError('This counter type did not establish complete inventory at bootstrap')
    if len(samples)<2:raise ValueError('At least two counter samples are needed')
    parsed=[]
    for row in samples:
        if row.get('exit_code')!=0:raise ValueError('An intermediate counter query failed')
        values=counters(row['stdout'])
        if set(values)!=expected:raise ValueError('A counter sample omits or adds an assigned GPU/link/direction')
        if row['end_unix_ns']<row['start_unix_ns']:raise ValueError('Invalid query timestamps')
        if parsed and any(values[k]<parsed[-1][k] for k in expected):
            raise ValueError('An intermediate counter decrease/reset makes this interval unusable')
        parsed.append(values)
    first,last=samples[0],samples[-1];a,b=parsed[0],parsed[-1]
    cadence=[(y['start_unix_ns']-x['start_unix_ns'])/1e9 for x,y in zip(samples,samples[1:])]
    if any(t<=0 for t in cadence):raise ValueError('Counter timestamps are not strictly increasing')
    elapsed=((last['start_unix_ns']+last['end_unix_ns'])-(first['start_unix_ns']+first['end_unix_ns']))/2e9
    if elapsed<=0:raise ValueError('No positive observation interval')
    rows=[{'gpu_uuid':k[0],'link':k[1],'direction':k[2],'begin_kib':a[k],'end_kib':b[k],'delta_bytes':(b[k]-a[k])*1024} for k in sorted(expected)]
    tx=sum(r['delta_bytes'] for r in rows if r['direction']=='Tx')
    rx=sum(r['delta_bytes'] for r in rows if r['direction']=='Rx')
    # Each endpoint was read somewhere inside its query. Report the duration range too.
    minimum_seconds=(last['start_unix_ns']-first['end_unix_ns'])/1e9
    maximum_seconds=(last['end_unix_ns']-first['start_unix_ns'])/1e9
    return {'first_query_ns':[first['start_unix_ns'],first['end_unix_ns']],
            'last_query_ns':[last['start_unix_ns'],last['end_unix_ns']],
            'sample_count':len(samples),'complete_counter_fields':len(expected),
            'sample_start_cadence_seconds':stats(cadence),
            'query_duration_ms':stats([(s['end_unix_ns']-s['start_unix_ns'])/1e6 for s in samples]),
            'elapsed_midpoint_seconds':elapsed,'elapsed_query_bound_seconds':[minimum_seconds,maximum_seconds],
            'tx_bytes_counted_once':tx,'rx_bytes_crosscheck':rx,
            'tx_GB_per_second':tx/elapsed/1e9,'rx_GB_per_second_crosscheck':rx/elapsed/1e9,
            'tx_GB_per_second_query_bounds':[tx/maximum_seconds/1e9,tx/minimum_seconds/1e9] if minimum_seconds>0 else None,
            'per_link':rows,'unit':'Decimal GB/s from documented KiB counters (1024 bytes per KiB)',
            'timestamp_limit':'Query-window rate bounds assume counters belong to their query intervals; nvidia-smi does not expose internal NVML cache lag.',
            'meaning':'Selected GPU-port egress/ingress deltas, not unique logical tensor bytes or exact phase bytes.'}


def window(rows,begin_ns,end_ns,expected):
    result={}
    for kind in ('d','r'):
        selected=sorted((r for r in rows if r.get('kind')==kind),key=lambda r:r['start_unix_ns'])
        before=[r for r in selected if r['end_unix_ns']<=begin_ns]
        after=[r for r in selected if r['start_unix_ns']>=end_ns]
        inner=[r for r in selected if r['start_unix_ns']>=begin_ns and r['end_unix_ns']<=end_ns]
        views={}
        for label,pair in [('inner',(inner[0],inner[-1]) if len(inner)>1 else None),
                           ('bracketing',(before[-1],after[0]) if before and after else None)]:
            if pair:
                interval=[r for r in selected if pair[0]['start_unix_ns']<=r['start_unix_ns']<=pair[1]['start_unix_ns']]
                try:
                    views[label]=delta(interval,expected[kind])
                    views[label]['first_query_relative_to_phase_start_ms']=[(t-begin_ns)/1e6 for t in views[label]['first_query_ns']]
                    views[label]['last_query_relative_to_phase_end_ms']=[(t-end_ns)/1e6 for t in views[label]['last_query_ns']]
                except ValueError as e:views[label]={'error':str(e)}
            else:views[label]={'unavailable':'Not enough counter samples around this window'}
        result[kind]=views
    return {'requested_window_ns':[begin_ns,end_ns],'counter_types':result,
            'limits':['Inner samples lie within the phase but omit its unsampled edges.',
                      'Bracketing samples include adjacent work outside the phase; their bytes are not exact phase bytes.',
                      'Every intermediate sample is checked against the active-link inventory and for counter decreases.',
                      'Queries read links sequentially; query-duration and cadence ranges are retained.',
                      'TX is counted once across selected GPU links. RX is a separate cross-check.',
                      'Payload and raw queries are staggered; their difference is not exact protocol overhead.',
                      'NVML cache timestamps are not exposed by this CLI; query-window timing bounds are conditional.',
                      'Raw counters include idle protocol traffic; no automatic background subtraction.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--show',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();show=json.loads(a.show.read_text());rows=[json.loads(line) for line in a.samples.open() if line.strip()]
    if show['status']!='completed':raise ValueError('Expected completed inference')
    expected=inventory(json.loads(a.inventory.read_text()),[g['uuid'] for g in show['gpus']])
    spans={s['name']:(int(s['start_unix_ms']*1e6),int((s['start_unix_ms']+s['ms'])*1e6)) for s in show['stages'] if s['name'] in ('condition_text','denoise','decode_video')}
    spans['gpu_receipt_span']=(min(g['start_us']*1000 for g in show['gpus']),max(g['end_us']*1000 for g in show['gpus']))
    steps=next((s for s in show.get('steps',[]) if s['name']=='denoise'),None)
    if steps and steps['count']==8 and len(steps['series'])==8:
        series=steps['series']
        for label,part in [('dense_steps_1_4',series[:4]),('sol_steps_5_8',series[4:])]:
            spans[label]=(int((part[0][0]-part[0][1])*1e6),int(part[-1][0]*1e6))
    result={'run':show['number'],'remote_execution_ms':show['execution_ms'],'inventory_source':str(a.inventory),
            'windows':{name:window(rows,*span,expected) for name,span in spans.items()}}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'run':show['number'],'windows':list(result['windows'])}))

if __name__=='__main__':main()
