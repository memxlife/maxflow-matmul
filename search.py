"""Synthetic shared-memory tile model; standard-library only."""
import argparse
import csv
import json
import math
from collections import deque
from itertools import product
from pathlib import Path


def max_flow(vertices, edges, source, sink):
    """Edmonds-Karp with explicit residual reverse edges."""
    residual = [[0.0] * vertices for _ in range(vertices)]
    for u, v, capacity in edges:
        residual[u][v] += capacity
    total = 0.0
    while True:
        parent = [-1] * vertices
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v, capacity in enumerate(residual[u]):
                if capacity > 0 and parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] < 0:
            return total
        amount = math.inf
        v = sink
        while v != source:
            u = parent[v]
            amount = min(amount, residual[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= amount
            residual[v][u] += amount
            v = u
        total += amount


def validate(c):
    for key in ('M', 'N', 'K', 's', 'W', 'T', 'S', 'R', 'r'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['r'] < 3:
        raise ValueError('r must be at least 3')
    for key in ('B_H', 'B_S', 'B_R', 'P'):
        if type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('m_values', 'n_values', 'k_values'):
        if not isinstance(c[key], list) or not c[key] or any(type(x) is not int or x <= 0 for x in c[key]):
            raise ValueError(f'{key} must be a nonempty list of positive integers')


def candidate(c, m, n, k):
    s, K, W = c['s'], c['K'], c['W']
    row = dict(m=m, n=n, k=k, threads=m*n, shared_bytes=s*k*(m+n), register_bytes=s*c['r']*m*n)
    reasons = []
    for dim, tile in (('M', m), ('N', n), ('K', k)):
        if c[dim] % tile:
            reasons.append(f'{dim} not divisible by tile')
    if n % W:
        reasons.append('n not a multiple of W')
    for field, limit in (('threads', 'T'), ('shared_bytes', 'S'), ('register_bytes', 'R')):
        if row[field] > c[limit]:
            reasons.append(f'{field} exceeds {limit}')
    row.update(legal=not reasons, rejection_reasons=reasons)
    if reasons:
        return row
    demands = {'H': s*(K*(m+n)+m*n), 'S': s*K*(m+n+m*n//W+m*n), 'R': s*(6*m*n*K+2*m*n+2*K*(m+n))}
    caps = {name: c['B_'+name]/value for name, value in demands.items()}
    caps['P'] = c['P']/(m*n*K)
    flow = max_flow(5, [(i, i+1, value) for i, value in enumerate(caps.values())], 0, 4)
    cut = min(caps.values())
    if not math.isclose(flow, cut, rel_tol=1e-12):
        raise AssertionError('maximum flow disagrees with analytical cut')
    row.update({f'D_{name}': value for name, value in demands.items()})
    row.update({f'c_{name}': value for name, value in caps.items()})
    row.update(flow_tiles_per_s=flow, analytical_tiles_per_s=cut, output_elements_per_s=m*n*flow,
               bottlenecks=[name for name, value in caps.items() if math.isclose(value, cut, rel_tol=1e-12)])
    return row


def search(c):
    validate(c)
    rows = [candidate(c, *shape) for shape in product(*[sorted(set(c[key])) for key in ('m_values', 'n_values', 'k_values')])]
    legal = sorted((r for r in rows if r['legal']), key=lambda r: (-r['output_elements_per_s'], r['m'], r['n'], r['k']))
    winners = [r for r in legal if math.isclose(r['output_elements_per_s'], legal[0]['output_elements_per_s'], rel_tol=1e-12)] if legal else []
    return dict(model='synthetic-baseline-v1', config=c, legal_count=len(legal), rejected_count=len(rows)-len(legal), winners=winners,
                rankings=legal, rejected=[r for r in rows if not r['legal']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('example.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = search(json.loads(args.config.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'rankings.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = result['rankings'] + result['rejected']
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir/'rankings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in rows)
    print(json.dumps({key: result[key] for key in ('legal_count', 'rejected_count', 'winners')}, indent=2))


if __name__ == '__main__':
    main()
