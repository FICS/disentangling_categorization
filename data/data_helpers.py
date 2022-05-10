import os
import gdown
import urllib.request
import hashlib
import zipfile
import shutil
import tarfile
import shlex
import io
import time
import sys
import subprocess
import signal

from tqdm import tqdm


class HashError(ValueError):
    pass


class PipelineError(ValueError):
    pass


def check_dirs(output, log=print):
    dirs = output.split('/')
    dirs_path = '/'.join(dirs[:-1])
    if not os.path.exists(dirs_path):
        os.makedirs(dirs_path)

        
# pip install gdown
def gdown_maybe(url, output, log=print):
    check_dirs(output)
    log(f"Checking if we need {output}")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        
        
def urllib_maybe(url, output, log=print):
    check_dirs(output)
    log(f"Checking if we need {output}")
    if not os.path.exists(output):
        log(f"Downloading {output}...")
        
        pb = None
        
        # Based on https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
        def reporthook(count, block_size, total_size):
            global pb
            global start_time
            global prev
            
            total_size_MB = int(total_size / (1024 * 1024))
            if count == 0:
                pb = tqdm(total=100)
                start_time = time.time()
                prev = 0
                return
            
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            old_prev = prev
            prev = int(count * block_size * 100 / total_size)
            update = prev - old_prev
            
            progress_size_MB = int(progress_size / (1024 * 1024))
            
            pb.update(update)
            pb.set_postfix(progress=f"...{progress_size_MB}/{total_size_MB}  MB, {speed} KB/s, {duration:.0f}s passed")
            
        urllib.request.urlretrieve(url, output, reporthook)

        
def print_hash(output, log=print):
    sha256 = hashlib.sha256()

    with open(output, 'rb') as f:
        data = f.read()
        sha256.update(data)

    got_hash = sha256.hexdigest()
    log(f'SHA256 hash: {got_hash}')
    
    
def check_hash(true, output, log=print):
    sha256 = hashlib.sha256()

    with open(output, 'rb') as f:
        data = f.read()
        sha256.update(data)
        
    got_hash = sha256.hexdigest()
    log(f'SHA256 hash: {got_hash}')
        
    if true != got_hash:
        raise HashError(f"The hash for {output} was incorrect! Please delete the corrupted file so I can try to re-download it.")

        
"""
Run shell command while streaming stdout+stderr to main proc and log file. 
Creates a new process group around the opened process, so it can kill all forked processes (e.g., torch data loaders)
"""
def command(args, log_path=None):
    if log_path is None:
        log_path = shlex.split(args)[0] + '.log'
        
    with open(log_path, 'a') as logf:
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, preexec_fn=os.setsid)
            # Use IOWrapper that will preserve tqdm carriage returns
            ostream = io.TextIOWrapper(p.stdout, encoding='utf-8', errors='strict', newline='')
            
            while True:
                output = ostream.readline()
                if len(output) == 0 and p.poll() is not None:
                    break
                if output:
                    # write to our buffers
                    sys.stdout.write(output)
                    logf.write(output)
                    
            rc = p.poll()
            return rc
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            raise KeyboardInterrupt
        except Exception as e:
            logf.write(e)
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            

"""
Quiet variant of command() for programs that tend to clog logs. 
v0: Just let stderr pipe to /dev/null. Popen will give us any python exceptions in parent. 
"""
def quiet_command(args):
    try:
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

        while True:
            if p.poll() is not None:
                break

        rc = p.poll()
        return rc
    except KeyboardInterrupt:
        p.kill()
        raise KeyboardInterrupt
    except Exception as e:
        p.kill()
        raise e