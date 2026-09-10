from dataclasses import replace
import json

import pytest

from rcias_ngas.rng import RNGStreams, NAMESPACES
from rcias_ngas.search.telemetry import RunState, Telemetry
from rcias_ngas.evaluation.bks import make_manifest, write_immutable
from rcias_ngas.evaluation.rpd import derive_rpd


def test_namespace_draw_and_key_independence():
    rng = RNGStreams('instance', 42)
    original = {n: [rng.stream(n, 'state', i).random() for i in range(8)] for n in NAMESPACES}
    for _ in range(19):
        rng.stream('target').random()
    assert original == {n: [rng.stream(n, 'state', i).random() for i in range(8)] for n in NAMESPACES}
    assert len({rng.seed(n) for n in NAMESPACES}) == len(NAMESPACES)
    changed = dict(original)
    changed['target'] = [rng.stream('target', 'new-state', i).random() for i in range(8)]
    assert changed['target'] != original['target']
    assert all(changed[n] == original[n] for n in NAMESPACES if n != 'target')
    with pytest.raises(ValueError):
        rng.stream('typo')


def test_matched_top1_sampling_downstream():
    rng = RNGStreams('instance', 3)
    chosen = rng.stream('portfolio').choices(['same'], k=1)[0]
    assert chosen == 'same'
    for name in ('repair', 'neighbor', 'acceptance'):
        sampled, top1 = rng.stream(name, 'it1'), rng.stream(name, 'it1')
        assert [sampled.random() for _ in range(10)] == [top1.random() for _ in range(10)]


def test_checkpoint_work_increases_without_new_best():
    telemetry = Telemetry(100)
    state = RunState(1, 0, 100, 100)
    telemetry.observe(1, state)
    state = replace(state, decoder_evals=30, iteration=4, best_makespan=90)
    telemetry.observe(70, state)
    telemetry.observe(75, state)
    state = replace(state, decoder_evals=70, neural_calls=2, iteration=9)
    telemetry.observe(99, state)
    result = telemetry.finish(100)
    three_quarter, final = result['budget_checkpoints'][-2:]
    assert three_quarter['best_makespan'] == final['best_makespan'] == 90
    assert three_quarter['decoder_evals'] == 30
    assert final['decoder_evals'] == 70
    assert result['last_best_time_sec'] == 70
    assert result['last_best_decoder_evals'] == 30


def test_checkpoints_do_not_backdate_completed_work():
    telemetry = Telemetry(10)
    telemetry.observe(3, RunState(1, 0, 100, 100))
    assert [p['decoder_evals'] for p in telemetry.checkpoints] == [0, 0]
    telemetry.observe(11, RunState(2, 1, 80, 80))
    result = telemetry.finish(11)
    assert result['budget_checkpoints'][-1]['decoder_evals'] == 1
    assert result['termination']['decoder_evals'] == 2
    with pytest.raises(ValueError):
        telemetry.observe(12, RunState())


def test_bks_update_preserves_raw_and_history(tmp_path):
    raw = {'instance_id': 'x', 'instance_sha256': 'hash', 'final_makespan': 100.,
           'feasible': True, 'raw_result_path': 'alns.json', 'raw_sha256': 'raw',
           'algorithm_id': 'ALNS', 'seed': 1}
    before = json.dumps(raw, sort_keys=True)
    v1 = make_manifest([raw], 1)
    old = derive_rpd([raw], v1)
    v2 = make_manifest([{**raw, 'final_makespan': 80, 'algorithm_id': 'NGAS',
                         'raw_result_path': 'ngas.json'}], 2, v1)
    assert derive_rpd([raw], v2)['rows'][0]['rpd_percent'] == 25
    assert derive_rpd([raw], v1) == old
    assert json.dumps(raw, sort_keys=True) == before
    write_immutable(tmp_path / 'bks.json', v1)
    with pytest.raises(FileExistsError):
        write_immutable(tmp_path / 'bks.json', v2)
    assert json.loads((tmp_path / 'bks.json').read_text()) == v1
    with pytest.raises(ValueError):
        make_manifest([{**raw, 'instance_sha256': 'drift'}], 2, v1)
