from __future__ import annotations

import inspect
from verifiers import load_environment

ENV_ID = "robocourier"  # важливо: без sutan/

def main():
    env = load_environment(ENV_ID)

    print("Loaded:", ENV_ID)
    print("Type:", type(env))
    print("Has reset:", hasattr(env, "reset"))
    print("Has step:", hasattr(env, "step"))
    print("Has action_space:", hasattr(env, "action_space"))
    print("Has observation_space:", hasattr(env, "observation_space"))

    # 1) reset()
    try:
        print("reset signature:", inspect.signature(env.reset))
    except Exception:
        pass

    reset_out = env.reset()
    print("\nreset() output type:", type(reset_out))
    print("reset() output:", reset_out)

    # 2) step() with sampled action
    if hasattr(env, "action_space"):
        a = env.action_space.sample()
        print("\nsampled action:", a)

        try:
            print("step signature:", inspect.signature(env.step))
        except Exception:
            pass

        step_out = env.step(a)
        print("\nstep() output type:", type(step_out))
        print("step() output:", step_out)
    else:
        print("\nNo action_space; cannot sample an action automatically.")

if __name__ == "__main__":
    main()

