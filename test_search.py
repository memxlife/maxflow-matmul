import json
import math
import unittest
from pathlib import Path
from search import candidate, max_flow, search, validate


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(Path(__file__).with_name('example.json').read_text())

    def test_general_network(self):
        self.assertEqual(max_flow(4, [(0,1,3),(0,2,2),(1,2,1),(1,3,2),(2,3,3)],0,3), 5)

    def test_all_chains(self):
        for row in search(self.c)['rankings']:
            self.assertEqual(row['flow_tiles_per_s'], min(row['c_'+x] for x in 'HSRP'))
            self.assertEqual(row['output_elements_per_s'], row['m']*row['n']*row['flow_tiles_per_s'])

    def test_independent_event_counts(self):
        c = dict(self.c, M=4, N=8, K=6, W=4)
        for k in (1,2,3,6):
            m,n=2,8
            h=shared=register=0
            for start in range(0,c['K'],k):
                for _ in range(m*k+k*n):
                    h+=1; shared+=1; register+=2
                for t in range(start,start+k):
                    for first in range(0,m*n,c['W']):
                        lanes=range(first,first+c['W'])
                        shared+=len({(lane//n,t) for lane in lanes})
                        shared+=len({(t,lane%n) for lane in lanes})
                        register+=6*c['W']
            h+=m*n; register+=2*m*n
            row=candidate(c,m,n,k)
            self.assertEqual([row['D_'+x] for x in 'HSR'],[4*h,4*shared,4*register])

    def test_scalar_chunked_matmul(self):
        A=[[i-t for t in range(6)] for i in range(4)]
        B=[[t+2*j for j in range(8)] for t in range(6)]
        reference=[[sum(A[i][t]*B[t][j] for t in range(6)) for j in range(8)] for i in range(4)]
        for k in (1,2,3,6):
            output=[[0]*8 for _ in range(4)]
            for base in range(0,4,2):
                for start in range(0,6,k):
                    a=[row[start:start+k] for row in A[base:base+2]]
                    b=B[start:start+k]
                    for lane in range(16):
                        i,j=divmod(lane,8)
                        for t in range(k):
                            output[base+i][j]+=a[i][t]*b[t][j]
            self.assertEqual(output,reference)

    def test_resources_and_domains(self):
        for config,shape,reason in [(dict(self.c,T=1),(4,32,1),'threads'),(dict(self.c,S=1),(4,32,1),'shared_bytes'),(dict(self.c,R=1),(4,32,1),'register_bytes'),(self.c,(3,32,1),'M not'),(self.c,(4,16,1),'n not'),(self.c,(4,32,3),'K not')]:
            row=candidate(config,*shape)
            self.assertFalse(row['legal'])
            self.assertTrue(any(reason in x for x in row['rejection_reasons']))
        self.assertEqual(search(dict(self.c,T=1))['winners'],[])

    def test_invalid_inputs(self):
        for key,value in [('M',0),('s',True),('r',2),('B_H',float('nan')),('P',-1),('k_values',[0])]:
            with self.assertRaises(ValueError):
                validate(dict(self.c,**{key:value}))

    def test_ties_and_area_ranking(self):
        result=search(self.c)
        self.assertEqual((result['legal_count'],result['rejected_count']),(37,38))
        self.assertEqual([(r['m'],r['n'],r['k']) for r in result['winners']],[(32,32,k) for k in (1,8,32,128)])
        small=candidate(self.c,4,32,1)
        large=candidate(self.c,32,32,1)
        self.assertGreater(small['flow_tiles_per_s'],large['flow_tiles_per_s'])
        self.assertLess(small['output_elements_per_s'],large['output_elements_per_s'])
        for m,n in {(r['m'],r['n']) for r in result['rankings']}:
            self.assertEqual(len({r['output_elements_per_s'] for r in result['rankings'] if (r['m'],r['n'])==(m,n)}),1)

    def test_example(self):
        row=candidate(self.c,16,32,8)
        self.assertEqual([row['D_'+x] for x in 'HSR'],[51200,589824,3248128])
        self.assertTrue(math.isclose(row['output_elements_per_s'],315258512.7187109,rel_tol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
