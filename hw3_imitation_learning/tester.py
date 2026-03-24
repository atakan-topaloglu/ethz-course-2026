"""
Script to test policy robustness across multiple environment seeds.
Usage: python scripts/eval_robustness.py --checkpoint checkpoints/single_cube/ex1.pt
"""

import subprocess
import re
import statistics
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Test checkpoint robustness across multiple seeds.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per seed")
    # You can add --multicube here if you want to reuse this for Ex3 later
    parser.add_argument("--multicube", action="store_true", help="Evaluate in multicube scene")
    args = parser.parse_args()

    # A list of completely arbitrary seeds to test against
    seeds = [10, 42, 99, 123, 777]
    success_rates = []

    print(f"Testing checkpoint: {args.checkpoint}")
    print(f"Episodes per seed:  {args.num_episodes}")
    print(f"Total test runs:    {len(seeds) * args.num_episodes}")
    print("-" * 50)

    for seed in seeds:
        # Build the command based on whether we are testing multicube or not
        cmd = [
            "python", "scripts/eval.py",
            "--checkpoint", args.checkpoint,
            "--num-episodes", str(args.num_episodes),
            "--headless",
            "--seed", str(seed)
        ]
        
        if args.multicube:
            cmd.append("--multicube")

        print(f"Running Seed {seed:3} ... ", end="", flush=True)
        
        # Run the evaluation script
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Scrape the terminal output for the final success rate string
        match = re.search(r"Success rate: \d+/\d+ \((\d+)%\)", result.stdout)
        
        if match:
            sr = float(match.group(1))
            success_rates.append(sr)
            print(f"{sr:>5.1f}%")
        else:
            print("FAILED TO PARSE")
            print(f"Error output:\n{result.stderr}")
            sys.exit(1)

    # Calculate final stats
    if success_rates:
        mean_sr = statistics.mean(success_rates)
        std_sr = statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
        
        print("\n" + "=" * 50)
        print("FINAL ROBUSTNESS RESULTS")
        print("=" * 50)
        print(f"Average Success Rate: {mean_sr:.2f}%")
        print(f"Standard Deviation:   {std_sr:.2f}%")
        print("=" * 50)
        
        if mean_sr >= 85.0 and (mean_sr - std_sr) >= 75.0:
            print("Verdict: ROCK SOLID. Ready for submission.")
        elif mean_sr >= 65.0:
            print("Verdict: OKAY, but variance might drop you a grade bracket.")
        else:
            print("Verdict: OVERFIT. You need more varied teleop data (DAgger).")

if __name__ == "__main__":
    main()