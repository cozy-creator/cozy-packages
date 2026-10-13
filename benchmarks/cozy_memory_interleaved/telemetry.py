"""External low-rate telemetry. NVML is initialized only by an authorized run."""
import json
import threading
import time
from pathlib import Path


class Sampler:
    def __init__(self, pid, gpu, path, interval=0.25):
        import psutil
        import pynvml
        self.psutil, self.n = psutil, pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.pid, self.path, self.interval = pid, Path(path), interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        n,h=self.n,self.handle
        memory=n.nvmlDeviceGetMemoryInfo(h); utilization=n.nvmlDeviceGetUtilizationRates(h)
        row={'monotonic_ns':time.perf_counter_ns(),'unix_ns':time.time_ns(),
             'gpu_used_bytes':memory.used,'gpu_free_bytes':memory.free,'gpu_total_bytes':memory.total,
             'gpu_util_percent':utilization.gpu,'memory_util_percent':utilization.memory,
             'temperature_c':n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU),
             'sm_clock_mhz':n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_SM),
             'memory_clock_mhz':n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM)}
        for key,call in {'power_mw':lambda:n.nvmlDeviceGetPowerUsage(h),
                         'pcie_tx_kb_s':lambda:n.nvmlDeviceGetPcieThroughput(h,n.NVML_PCIE_UTIL_TX_BYTES),
                         'pcie_rx_kb_s':lambda:n.nvmlDeviceGetPcieThroughput(h,n.NVML_PCIE_UTIL_RX_BYTES)}.items():
            try:row[key]=call()
            except n.NVMLError:row[key]=None
        try:
            row['compute_processes']=[{'pid':p.pid,'gpu_bytes':p.usedGpuMemory} for p in n.nvmlDeviceGetComputeRunningProcesses(h)]
        except n.NVMLError:row['compute_processes']=[]
        processes=[]
        try:
            parent=self.psutil.Process(self.pid)
            for p in [parent,*parent.children(recursive=True)]:
                try:
                    cpu=p.cpu_times(); mem=p.memory_info(); io=p.io_counters()
                    processes.append({'pid':p.pid,'cpu_user_s':cpu.user,'cpu_system_s':cpu.system,'rss_bytes':mem.rss,
                                      'read_bytes':io.read_bytes,'write_bytes':io.write_bytes,'read_chars':io.read_chars,'write_chars':io.write_chars})
                except self.psutil.Error:pass
        except self.psutil.Error:pass
        row['processes']=processes
        return row

    def run(self):
        with self.path.open('w') as out:
            while not self.stop_event.is_set():
                try:row=self.sample()
                except Exception as e:row={'monotonic_ns':time.perf_counter_ns(),'error':str(e)}
                out.write(json.dumps(row)+'\n');out.flush()
                self.stop_event.wait(self.interval)

    def start(self):self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:self.thread.join()
        self.n.nvmlShutdown()
