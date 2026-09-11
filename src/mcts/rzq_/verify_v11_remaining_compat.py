"""Cross-check physical rollout keys against stored path and rank trace."""
from collections import defaultdict
import json
from audit_environment import HERE
from mcts.rzq_.phase_restart import load_task
p=HERE/'runs/class8_v11';out=p/'remaining_compat_last300'
t=load_task((p/'final_snapshot.pkl').read_bytes())
ns={n.creation_index:n for n in t.state.nodes.values()}
groups=defaultdict(list)
for line in (p/'decisions.jsonl').open():
 e=json.loads(line)
 if 1095<=e.get('iteration',0)<=1394:groups[e['iteration']].append(e)
steps=0
for it,es in groups.items():
 bp=next(e for e in es if e['event']=='real_visit_backpropagation')
 n=ns[bp['edges'][-1]['child_id']];key=tuple(n.key);path=list(n.path)
 for e in es:
  if e['event']!='rollout_choice':continue
  assert e['current_path']==path
  assert t.state.scorer.affine_rank(key)==e['current_rank']
  assert key[e['chosen_action']]==0
  key=list(key);key[e['chosen_action']]=1;key=tuple(key)
  path.append(e['chosen_action'])
  assert t.state.scorer.affine_rank(key)==e['chosen_new_rank']
  steps+=1
 final=t.state.scorer.compatibility.counts(key)
 assert all(final[c]==0 for c in (9,11,25))
r=json.loads((out/'data.json').read_text())
for c in ('9','11','25','any'):
 assert sum(r['rollout_summary'][c].get(k,0) for k in ('sample_omitted_safe_actions','top_k_removed_safe_actions','softmax_chose_incompatible'))==len(r['rollout_eliminations'][c])
audit={'passed':True,'iterations':len(groups),'rollout_steps_rank_and_path_verified':steps,
       'actual_closed_tree_keys_used':True,'all_final_target_compats_zero':True,'first_loss_categories_reconcile':True}
(out/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
print(audit)
