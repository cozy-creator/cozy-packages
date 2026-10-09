from pathlib import Path
import json,tempfile,unittest
from parse_transport_logs import collect
from transport_control import COMMON,SEALED,PREFIX

class Parser(unittest.TestCase):
    def fixture(self, mode='host', wrong=False,missing=False):
        lines=[]
        for rank in range(4):
            pid=100+rank
            lines.append(PREFIX+json.dumps({'pid':pid,'mode':mode,'environment':{**COMMON,'NCCL_P2P_DISABLE':'1' if mode=='host' else '0'},'unchanged_seal':SEALED,'before_process_group':True}))
            lines.append(f'host:{pid}:900 [{rank}] NCCL INFO NCCL version 2.30.7+cuda13.0')
            for peer in range(4):
                if rank==peer or (missing and rank==0 and peer==1):continue
                transport='P2P/CUMEM' if mode=='peer' or wrong else 'SHM/direct'
                lines.append(f'host:{pid}:900 [{rank}] NCCL INFO Channel 00/1 : {rank}[{rank}] -> {peer}[{peer}] via {transport}')
        return lines
    def parse(self,lines,mode='host'):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stdout.log';p.write_text('\n'.join(lines))
            return collect({'number':1,'status':'completed','degree':4,'gpus':[{'pid':100+r,'gpu':r} for r in range(4)]},[p],mode)
    def test_both_routes(self):
        for mode in ('peer','host'):
            r=self.parse(self.fixture(mode),mode);self.assertTrue(r['complete'],r['issues']);self.assertEqual(len(r['directed_physical_gpu_links']),12)
    def test_flags_alone_not_accepted(self):
        r=self.parse([s for s in self.fixture() if 'Channel' not in s]);self.assertFalse(r['complete'])
    def test_wrong_transport_refused(self):self.assertFalse(self.parse(self.fixture(wrong=True))['complete'])
    def test_missing_pair_refused(self):self.assertFalse(self.parse(self.fixture(missing=True))['complete'])
    def test_foreign_pid_cannot_supply_missing_pair(self):
        lines=self.fixture(missing=True)+['host:999:900 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct']
        self.assertFalse(self.parse(lines)['complete'])
    def test_unfinished_or_failed_run_refused(self):
        for status in ('in_progress','failed'):
            with self.assertRaises(ValueError):
                collect({'number':1,'status':status,'degree':4,'gpus':[{'pid':100+r,'gpu':r} for r in range(4)]},[],'host')
    def test_two_rank_communicator_labels_do_not_fake_global_pair(self):
        lines=self.fixture(missing=True)
        lines.append('host:100:900 [0] NCCL INFO Channel 00/1 : 0[0] -> 1[3] via SHM/direct')
        self.assertFalse(self.parse(lines)['complete'])
    def test_duplicate_control_refused(self):
        lines=self.fixture();self.assertFalse(self.parse(lines+[lines[0]])['complete'])

if __name__=='__main__':unittest.main()
