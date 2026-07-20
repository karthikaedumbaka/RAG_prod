import json
from pathlib import Path

try:
    from .config import PipelineConfig
    from .logger import setup_logger
except ImportError:
    from config import PipelineConfig
    from logger import setup_logger

log = setup_logger("checkpoint")

class CheckpointManager:
    def __init__(self, config: PipelineConfig):
        self.state_file = Path(config.output_dir) / config.checkpoint_file
        self.state = self._load_state()
        
    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                log.info(f" Loaded checkpoint: {len(state.get('completed_batches', []))} batches completed")
                return state
        return {"completed_batches": []}
        
    def _save_state(self):
        # Atomic write to prevent corruption
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.state_file)
        
    def is_completed(self, batch_name: str) -> bool:
        return batch_name in self.state["completed_batches"]
        
    def mark_completed(self, batch_name: str):
        if batch_name not in self.state["completed_batches"]:
            self.state["completed_batches"].append(batch_name)
            self._save_state()
            log.debug(f" Checkpoint: {batch_name} marked complete")