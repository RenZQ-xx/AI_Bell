"""Independent end-of-run audit for v11 stopping and restart accounting."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text())

def verify(path):
    result=read(path/'result.json');known=set(result['meta']['pre_discovered_class_ids'])
    iteration_count=0;last_discovery=0;new_events=[];resets=[];real_updates=0;special=0
    last_backprop=-1;sequence=[];expansions=0
    with (path/'decisions.jsonl').open() as f:
        for line in f:
            e=json.loads(line);event=e['event'];i=e.get('iteration',0)
            if event=='iteration_start':
                iteration_count+=1
                assert i==iteration_count
                assert e['root_visits']==i-1
            if event=='real_visit_backpropagation':
                real_updates+=1;last_backprop=i
            if event=='new_class_discovery':
                assert last_backprop==i
                assert e['class'] not in known
                known.add(e['class']);new_events.append((i,e['class']));last_discovery=i
                assert known.issubset(set(e['global_classes']))
            if event=='restart':
                assert last_backprop==i
                assert all(c in known for c in e['classes'])
                assert set(e['classes'])=={c for j,c in new_events if j==i}
                before=e['before'];after=e['after']
                assert before['nodes']==after['nodes'] and before['edges']==after['edges']
                for a,b in zip(before['node_statistics'],after['node_statistics']):
                    assert a['id']==b['id'] and a['total_visits']==b['total_visits']
                    assert b['phase_visits']==0 and b['stage_slots_used']==0 and b['stage_capacity']==6
                    assert a['next_bucket']==b['next_bucket'] and a['bucket_expansion_counts']==b['bucket_expansion_counts']
                    for x,y in zip(a['edges'],b['edges']):
                        assert y['q']==0 and y['edge_phase_visits']==0
                        assert x['exploration']==y['exploration'] and x['edge_total_visits']==y['edge_total_visits']
                resets.append(i)
            if event=='expand_action':
                expansions+=1
                assert e['children_before']<e['effective_limit']
                if e['stage_capacity'] is not None:assert e['stage_slots_used']<e['stage_capacity']
            if event=='sealed_selection':
                special+=1
                assert e['slots_used_after']==e['slots_used_before']+1<=e['capacity']
                assert all(c['edge_phase_visits']==0 for c in e['candidates'])
                assert e['chosen_action']==max(e['candidates'],key=lambda c:c['exploration'])['action']
                assert not ({c['action'] for c in e['candidates']} & set(e['excluded_actions']))
            if event=='ucb_choice':
                for c in e['candidates']:
                    assert isinstance(c['edge_phase_visits'],int)
                    assert math.isclose(c['exploration'],1.4*c['prior']*max(1,e['parent_total_visits'])**.5/(1+c['visits']),rel_tol=1e-12,abs_tol=1e-12)
    assert iteration_count==real_updates==result['iterations']
    assert len(resets)==result['restart_count'] and len(resets)==len(set(resets))
    assert last_discovery==result['last_new_iteration']
    assert iteration_count-last_discovery==result['stale_iterations']==300
    assert known==set(result['global_classes'])
    assert sum(s['length'] for s in result['stages'])==iteration_count
    assert result['stages'][-1]['length']==300
    incoming_total=Counter();incoming_phase=Counter()
    for n in result['tree']['node_statistics']:
        for e in n['edges']:
            incoming_total[e['child_id']]+=e['edge_total_visits'];incoming_phase[e['child_id']]+=e['edge_phase_visits']
    for n in result['tree']['node_statistics']:
        if n['id']!=result['tree']['root_id']:
            assert n['total_visits']==incoming_total[n['id']]
            assert n['phase_visits']==incoming_phase[n['id']]
    smoke=path.parent/'class8_v11_smoke20/decisions.jsonl'
    short=[json.loads(x) for x in smoke.read_text().splitlines()]
    prefix=[]
    with (path/'decisions.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r.get('iteration',0)>20:break
            prefix.append(r)
    assert short==prefix
    audit={'passed':True,'iterations':iteration_count,'last_new':last_discovery,'stale':300,
           'restarts':len(resets),'new_classes':new_events,'special_selections':special,'expansions':expansions,
           'integer_counts_match_incoming_edges':True,'restart_preserves_total_visits_and_exploration':True,
           'each_restart_grants_six_slots':True,'global_class_updates_and_event_order_checked':True,
           'formal_first20_matches_smoke':True}
    (path/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);verify(p.parse_args().path)

