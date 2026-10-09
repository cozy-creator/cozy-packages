import unittest
from analyze_links import counters,delta,inventory,window

class Links(unittest.TestCase):
    expected={('GPU-a',0,'Tx'),('GPU-a',0,'Rx')}
    def row(self,t,value,kind='d'):
        return {'kind':kind,'exit_code':0,'start_unix_ns':int(t*1e9),'end_unix_ns':int((t+.1)*1e9),'stdout':f'GPU 0: H100 (UUID: GPU-a)\n Link 0: Data Tx: {value} KiB\n Link 0: Data Rx: {value} KiB\n'}
    def test_tx_is_not_doubled(self):
        d=delta([self.row(1,100),self.row(3,300)],self.expected)
        self.assertEqual(d['tx_bytes_counted_once'],200*1024);self.assertEqual(d['rx_bytes_crosscheck'],200*1024)
        self.assertAlmostEqual(d['elapsed_query_bound_seconds'][0],1.9)
        self.assertEqual(d['sample_start_cadence_seconds']['mean'],2)
    def test_intermediate_reset_refused_even_when_final_recovers(self):
        with self.assertRaises(ValueError):delta([self.row(1,100),self.row(3,99),self.row(5,200)],self.expected)
    def test_same_partial_inventory_refused(self):
        full=self.expected|{('GPU-b',0,'Tx'),('GPU-b',0,'Rx')}
        with self.assertRaises(ValueError):delta([self.row(1,100),self.row(3,200)],full)
    def test_intermediate_failure_refused(self):
        bad=self.row(2,150);bad['exit_code']=1
        with self.assertRaises(ValueError):delta([self.row(1,100),bad,self.row(3,200)],self.expected)
    def test_inner_and_bracketing_are_distinct(self):
        d=window([self.row(t,t*100) for t in (0,2,4,6,8)],int(1e9),int(7e9),{'d':self.expected,'r':self.expected})['counter_types']['d']
        self.assertEqual(d['inner']['tx_bytes_counted_once'],400*1024)
        self.assertEqual(d['bracketing']['tx_bytes_counted_once'],800*1024)
        self.assertEqual(d['inner']['first_query_relative_to_phase_start_ms'],[1000,1100])
    def test_inventory_binds_assigned_gpu_and_active_links(self):
        receipt={'gpu_uuids':['GPU-a'],'counter_feasibility':{'status':{'stdout':'GPU 0: H100 (UUID: GPU-a)\n Link 0: 26.562 GB/s\n'},'data':self.row(1,100),'raw':self.row(1,100)}}
        self.assertEqual(inventory(receipt,['GPU-a'])['d'],self.expected)
        with self.assertRaises(ValueError):inventory(receipt,['GPU-b'])

if __name__=='__main__':unittest.main()
