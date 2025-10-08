#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import signal
import logging
import argparse
import subprocess
import threading
import socket
import requests
from queue import Queue
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"gpu_scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TEST_PROMPTS = ["What is the meaning of life?", "Tell me a joke.", "What is the capital of France?"]
class GPURessourceManager:
    def __init__(self, total_gpus):
        self.total_gpus = total_gpus
        self.available_gpus = list(range(total_gpus))  # GPU ID starts from 0
        self.lock = threading.Lock()

    def allocate_gpus(self, num_gpus):
        with self.lock:
            if len(self.available_gpus) >= num_gpus:
                allocated_gpus = self.available_gpus[:num_gpus]
                self.available_gpus = self.available_gpus[num_gpus:]
                logger.info(f"Allocate GPU: {allocated_gpus}")
                return allocated_gpus
            else:
                # Insufficient resources
                logger.warning(f"Insufficient GPU resources, requested {num_gpus}, available {len(self.available_gpus)}")
                return None

    def release_gpus(self, gpus):
        """Release GPU resources"""
        with self.lock:
            self.available_gpus.extend(gpus)
            self.available_gpus.sort()  # Keep GPU IDs in order
            logger.info(f"Released GPU: {gpus}")

    def get_available_count(self):
        """Get the number of available GPUs"""
        with self.lock:
            return len(self.available_gpus)


class TaskProcessor:
    """Task processor"""
    def __init__(self, task_queue, result_queue, gpu_manager, max_parallel_tasks):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gpu_manager = gpu_manager
        self.max_parallel_tasks = max_parallel_tasks
        self.active_processes = {}  # Track active processes
        self.lock = threading.Lock()
        self.running = True
        
    def start(self):
        """Start the task processor"""
        self.monitor_thread = threading.Thread(target=self._monitor_tasks)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_tasks(self):
        """Monitor task queue and start processes - Fix deadlock issue"""
        while self.running:
            # 1. First clean up completed processes and release GPU resources
            self._cleanup_finished_processes()
            
            # 2. Check if new tasks can be started
            task_to_start = None
            allocated_gpus = None
            
            # Acquire lock and check conditions, but don't hold lock while executing _start_task_process
            with self.lock:
                if len(self.active_processes) < self.max_parallel_tasks and not self.task_queue.empty():
                    # Try to obtain sufficient GPU resources
                    task = self.task_queue.queue[0]  # Look at the first task without removing it
                    num_gpus = task.get('gpu_count', 1)
                    allocated_gpus = self.gpu_manager.allocate_gpus(num_gpus)
                    
                    if allocated_gpus:
                        # Successfully allocated GPU, prepare to start task
                        task_to_start = self.task_queue.get()  # Remove task from queue
            
            # Key fix: Start task outside the lock to avoid deadlock
            if task_to_start and allocated_gpus:
                self._start_task_process(task_to_start, allocated_gpus)
            logger.info(f"active_processes: {len(self.active_processes)}")
            
            # 3. Short sleep to avoid high CPU usage
            time.sleep(1)
            
            # 4. Check if all tasks are completed
            with self.lock:
                all_tasks_done = self.task_queue.empty() and len(self.active_processes) == 0
                
            if all_tasks_done:
                break

    def _wait_for_port(self, host, port, timeout=3600, interval=10):
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=5):
                    logger.info(f"Port {port} is open")
                    return True
            except (socket.timeout, ConnectionRefusedError):
                logger.info(f"Port {port} is not open, waiting {interval} seconds before retrying")
                time.sleep(interval)
        logger.error(f"Port {port} did not open within {timeout} seconds")

        return False

    def _request_server(self, host, port, model_path, timeout=3600):
        # Check if port is open
        if not self._wait_for_port(host, port, timeout):
            logger.error(f"Port {port} is not open, cannot request server")
            return False
        
        """Request model service"""
        base_url = f'http://127.0.0.1:{port}'
        for prompt in TEST_PROMPTS:
            success = False
            url = f'{base_url}/v1/completions'
            data = {
                "model": model_path,
                "prompt": prompt,
                "max_tokens": 32,
            }
            try:
                response = requests.post(url, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Request successful, model: {model_path}, prompt: {prompt}, response: {result}")
                success = True
            except requests.RequestException as e:
                logger.error(f"Request failed, model: {model_path}, prompt: {prompt}, error: {e}")

            if success:
                continue

            # Try chat/completions
            url = f'{base_url}/v1/chat/completions'
            data = {
                "model": model_path,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
            }
            try:
                response = requests.post(url, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Request successful, model: {model_path}, prompt: {prompt}, response: {result}")
                success = True
            except requests.RequestException as e:
                logger.error(f"Request failed, model: {model_path}, prompt: {prompt}, error: {e}")
            
    def _start_task_process(self, task, allocated_gpus):
        """Start task process - Fix deadlock issue"""
        task_id = task.get('id', f"task_{int(time.time())}")
        
        # Create log file
        log_file = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Prepare environment variables
        env = os.environ.copy()
        if 'env' in task:
            env.update(task['env'])
        
        # Set CUDA_VISIBLE_DEVICES environment variable
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, allocated_gpus))
        
        logger.info(f"Starting task: {task_id}, GPU: {allocated_gpus}, command: {task['command']}")
        
        # Linux platform optimization: Use shell=True for more stability on Linux
        process = subprocess.Popen(
            task['command'],
            shell=True,
            env=env,
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # Linux-specific: Set process group for easier signal handling later
        )
        
        # Record active process information - Still need to acquire lock here, but now called outside the lock
        with self.lock:
            self.active_processes[process.pid] = {
                'process': process,
                'task_id': task_id,
                'allocated_gpus': allocated_gpus,
                'log_file': log_file,
                'start_time': time.time(),
                'timeout': task.get('timeout', 10)  # Timeout period (seconds)
            }
        
        # Issue fix: Put server requests in a separate thread to avoid blocking main process
        if 'model_path' in task:
            server_check_thread = threading.Thread(
                target=self._check_server_in_background, 
                args=(task, process.pid)
            )
            server_check_thread.daemon = True
            server_check_thread.start()
    
    def _check_server_in_background(self, task, pid):
        port = task['port']
        model_path = task['model_path']
        host = '127.0.0.1'
        timeout = task['timeout']
        """Check server status in background thread"""
        self._request_server(host, port, model_path, timeout)
        
        # Note: The logic to force terminate the process here may need re-evaluation
        # If force termination is indeed needed, first check if the process is still running
        with self.lock:
            if pid in self.active_processes and self.active_processes[pid]['process'].poll() is None:
                logger.info(f"Task is still running, PID: {pid}, force terminating...")
                
                try:
                    # First try to gracefully terminate with SIGTERM signal
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    logger.info(f"Sent SIGTERM signal to process group, PID: {pid}")
                    
                    # Wait for process to terminate on its own (max 5 seconds)
                    wait_start_time = time.time()
                    while time.time() - wait_start_time < 5:
                        if self.active_processes[pid]['process'].poll() is not None:
                            logger.info(f"Process terminated normally, PID: {pid}")
                            break
                        time.sleep(0.5)
                    
                    # If process still hasn't terminated, force terminate with SIGKILL
                    if self.active_processes[pid]['process'].poll() is None:
                        logger.warning(f"Process did not terminate within expected time, preparing to send SIGKILL signal, PID: {pid}")
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        logger.warning(f"Sent SIGKILL signal to process group, PID: {pid}")
                        
                        # Wait for force termination to complete
                        try:
                            self.active_processes[pid]['process'].wait(timeout=2)
                            logger.info(f"Process force terminated, PID: {pid}")
                        except subprocess.TimeoutExpired:
                            logger.error(f"Force terminating process timed out, PID: {pid}")
                            # Even if timed out, we continue processing as the process might have terminated but wait call timed out
                except ProcessLookupError:
                    logger.info(f"Process no longer exists: {pid}")
                except PermissionError:
                    logger.error(f"No permission to terminate process: {pid}")
                except OSError as e:
                    logger.error(f"OS error occurred while terminating process: {pid}, error: {e}")
                except Exception as e:
                    logger.error(f"Unknown error occurred while terminating process: {pid}, error: {e}")
                # finally:
                #     # Ensure GPU resources are released
                #     if pid in self.active_processes:
                #         gpus = self.active_processes[pid]['allocated_gpus']
                #         self.gpu_manager.release_gpus(gpus)
                #         logger.info(f"Released GPU resources occupied by process: {gpus}")
            else:
                logger.info(f"Process does not need termination (already completed or non-existent), PID: {pid}")

    def _cleanup_finished_processes(self):
        """Clean up completed processes and release resources - Linux optimized version"""
        with self.lock:
            pids_to_remove = []
            
            for pid, info in self.active_processes.items():
                # Check if process is completed
                retcode = info['process'].poll()
                if retcode is not None:
                    # Process completed
                    runtime = time.time() - info['start_time']
                    logger.info(f"Task completed: {info['task_id']}, PID: {pid}, return code: {retcode}, runtime: {runtime:.2f} seconds")
                    
                    # Release GPU resources
                    self.gpu_manager.release_gpus(info['allocated_gpus'])
                    
                    # Mark for removal
                    pids_to_remove.append(pid)
                elif info['timeout'] is not None:
                    # Check if timed out
                    if time.time() - info['start_time'] > info['timeout']:
                        logger.warning(f"Task timed out: {info['task_id']}, PID: {pid}, ran for {time.time() - info['start_time']:.2f} seconds, terminating...")
                        
                        # Linux platform optimization: Try SIGTERM first, then SIGKILL
                        try:
                            # Send signal to the entire process group
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                            # Wait for some time, force terminate if still running
                            info['process'].wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Force terminating task: {info['task_id']}, PID: {pid}")
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        
                        # Release GPU resources
                        self.gpu_manager.release_gpus(info['allocated_gpus'])
                        
                        # Mark for removal
                        pids_to_remove.append(pid)
            
            # Remove completed processes
            for pid in pids_to_remove:
                del self.active_processes[pid]

    def stop(self):
        """Stop task processor - Linux optimized version"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        # Terminate all active processes
        with self.lock:
            for pid, info in self.active_processes.items():
                logger.info(f"Terminating task: {info['task_id']}, PID: {pid}")
                try:
                    # Linux platform optimization: Send signal to process group
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    # Wait for some time, force terminate if still running
                    info['process'].wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        logger.warning(f"Force terminating task: {info['task_id']}, PID: {pid}")
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        logger.info(f"Process no longer exists: {pid}")
                
                # Release GPU resources
                self.gpu_manager.release_gpus(info['allocated_gpus'])
            
            self.active_processes.clear()


class GPUScheduler:
    """GPU task scheduler"""
    def __init__(self, total_gpus, max_parallel_tasks):
        self.gpu_manager = GPURessourceManager(total_gpus)
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.task_processor = TaskProcessor(
            self.task_queue, 
            self.result_queue, 
            self.gpu_manager, 
            max_parallel_tasks
        )
        
        # Register signal handling - Linux optimization
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        self.running = False

    def _handle_signal(self, signum, frame):
        """Handle signals - Linux optimization"""
        logger.info(f"Received signal {signum} ({signal.Signals(signum).name}), stopping scheduler...")
        self.stop()
        sys.exit(0)

    def add_task(self, task):
        """Add task to queue"""
        if not isinstance(task, dict):
            raise TypeError("Task must be a dictionary type")
        
        # Validate required task fields
        required_fields = ['command']
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task missing required field: {field}")
        
        # Set default values
        if 'gpu_count' not in task:
            task['gpu_count'] = 1
        if 'id' not in task:
            task['id'] = f"task_{int(time.time())}_{self.task_queue.qsize()}"
        
        self.task_queue.put(task)
        logger.info(f"Added task: {task['id']}")

    def add_tasks_from_file(self, file_path):
        """Load task list from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            common_env = json_data.get('common_env', {})
            w8a8_env = json_data.get('w8a8_env', {})
            tasks = json_data.get('tasks', [])
            if isinstance(tasks, list):
                port = 8000
                for task in tasks:
                    task['port'] = port
                    task['command'] = f"vllm serve {task['model_path']} {task['command']} --port {task['port']}"
                    task['env'] = task.get('env', {})
                    task['env'] = common_env | task['env']
                    task_env = task.get('ext_env', {})
                    if task_env == 'w8a8_env':
                        task['env'] = w8a8_env | task['env']
                    print(task)
                    self.add_task(task)
                    port += 1
                logger.info(f"Loaded {len(tasks)} tasks from file {file_path}")
            else:
                logger.error(f"Task format in file {file_path} is incorrect")
        except Exception as e:
            logger.error(f"Failed to load task file: {str(e)}")

    def start(self):
        """Start scheduler - Linux optimized version"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        logger.info("Starting GPU task scheduler")
        
        # Start task processor
        self.task_processor.start()
        
        # Wait for all tasks to complete
        try:
            while self.running:
                # Check if all tasks are completed
                with self.task_processor.lock:
                    all_tasks_done = self.task_queue.empty() and len(self.task_processor.active_processes) == 0
                
                if all_tasks_done:
                    break
                
                # Short sleep to avoid high CPU usage
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping scheduler...")
            self.stop()
        
        logger.info("All tasks completed, scheduler exiting")

    def stop(self):
        """Stop scheduler"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping GPU task scheduler")
        
        # Stop task processor
        self.task_processor.stop()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Linux platform GPU task scheduler')
    parser.add_argument('--total-gpus', type=int, default=8, help='Total number of available GPUs')
    parser.add_argument('--max-parallel', type=int, default=4, help='Maximum number of parallel tasks')
    parser.add_argument('--tasks-file', type=str, help='Task file path (JSON format)')
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = GPUScheduler(args.total_gpus, args.max_parallel)
    
    # If task file is provided, load tasks
    if args.tasks_file:
        scheduler.add_tasks_from_file(args.tasks_file)
    else:
        # Sample tasks
        sample_tasks = [
            {
                'id': 'task_1',
                'command': 'vllm serve /home/jovyan/models/Qwen/Qwen3-0.6B/',
                'model_path': '/home/jovyan/models/Qwen/Qwen3-0.6B/',
                'gpu_count': 2,
                'timeout': 300
            },
        ]
        
        port = 8000
        for task in sample_tasks:
            task['port'] = port
            task['command'] = f"{task['command']} --port {task['port']}"
            scheduler.add_task(task)
            port += 1
    
    # Start scheduler
    scheduler.start()


if __name__ == '__main__':
    main()