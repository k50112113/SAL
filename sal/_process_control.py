import subprocess
import signal
import time
import os

def wait_all_proc_end(proc_list, logfile_path, wait_interval = 60, wall_time = 86400):
    total_wait_time = 0
    end_code = 0
    proc_end = False
    while proc_end == False:
        time.sleep(wait_interval)
        proc_end = True
        for proc in proc_list:
            print(f"proc {proc.pid} ", end = "")
            if proc.poll() == None:
                print("running", flush = True)
                proc_end = False
            elif proc.poll() == 0:
                print("ended ", flush = True)
            else:
                print(f"ended with non-zero exit code: {proc.poll()}", flush = True)
                end_code = 1
        total_wait_time += wait_interval
        if total_wait_time > wall_time or os.path.isfile("salend"):
            write_log("Wall time limit reached or force terminate", logfile_path)
            for proc in proc_list:
                if proc.poll() == None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM) 
                    write_log(f"proc {proc.pid} killed", logfile_path)
                    end_code = 1
            break
        print(f"Current wall time: {total_wait_time}", flush = True)
    return end_code

def write_log(message, logfile_path):
    os.system(f"echo {message} >> {logfile_path}")
    print(message, flush = True)

def run_command(commandline, wait = False):
    proc = subprocess.Popen(commandline, shell=True, preexec_fn = os.setsid)
    if wait: proc.wait() 
    return proc

def go_to_path(target_path):
    os.chdir(target_path)

def next_cuda_card(idx, cuda_card_list):
    next_idx = (idx + 1) % len(cuda_card_list)
    return next_idx, cuda_card_list[next_idx]

class Clock:
    def __init__(self):
        self.reset_timer()

    def get_dt(self):
        self.time2 = time.perf_counter()
        dt = self.time2-self.time1
        self.time1 = self.time2
        return dt

    def reset_timer(self):
        self.time1 = time.perf_counter()
        self.time2 = None