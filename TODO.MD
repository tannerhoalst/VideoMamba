Summary of the Task

Find the selective_scan call: This is the math kernel. It almost always supports an initial_state argument, but the wrapper classes usually ignore it.

Wire it up: You are just "plumbing" the variable ssm_state from the top-level forward() down to that kernel, and then passing the result back up.

Why this is crucial: If you don't do this, your depth map will "hiccup" or "jump" every 12 frames because the model's memory gets wiped clean. With this change, the model effectively remembers the entire video history.