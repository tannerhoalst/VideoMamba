Summary of the Task

Find the selective_scan call: This is the math kernel. It almost always supports an initial_state argument, but the wrapper classes usually ignore it.

Wire it up: You are just "plumbing" the variable ssm_state from the top-level forward() down to that kernel, and then passing the result back up.

Why this is crucial: If you don't do this, your depth map will "hiccup" or "jump" every 12 frames because the model's memory gets wiped clean. With this change, the model effectively remembers the entire video history.



________________________________________________________________________

• Wired ssm_state through the VideoMamba forward path into the selective scan call and
  back out, using a local Mamba wrapper that passes initial_state when supported (with a
  reference fallback for correctness). models/videomamba/videomamba.py now
  accepts optional ssm_state, threads it per-layer, and returns updated state when
  provided, while default callers remain unchanged.

  Details:

  - Added models/videomamba/mamba_simple.py with ssm_state support and
  fast-path disabled when state is needed.
  - Updated models/videomamba/videomamba.py to pass per-layer state and return
  it when requested; supports list/tuple or dict (e.g., from allocate_inference_cache).
