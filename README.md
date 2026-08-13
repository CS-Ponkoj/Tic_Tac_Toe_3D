# 3D Tic-Tac-Toe with Minimax and Alpha-Beta Pruning

> A Python implementation of 2D and 3D Tic-Tac-Toe, centered on a 3 × 3 × 3 game engine that selects moves with minimax search and alpha-beta pruning.

## Overview

A Python implementation of 2D and 3D Tic-Tac-Toe, centered on a 3 × 3 × 3 game engine that selects moves with minimax search and alpha-beta pruning. This repository preserves the implementation, source data or supporting artifacts, and the original project outputs so the work can be reviewed and reproduced.

## Motivation

Three-dimensional Tic-Tac-Toe expands a familiar game into a larger state space. This project explores how board representation, win detection, heuristic evaluation, and search-depth control work together in a compact adversarial game.

## Goal and Research Question

**Goal:** Build a testable 3D game-state model and an automated move-selection loop for a 27-cell board.

**Question:** How can minimax search with alpha-beta pruning select competitive moves while keeping a 3D Tic-Tac-Toe search manageable?

## Technical Approach

1. Represent the board as a NumPy array with shape 3 × 3 × 3.
2. Map sequential moves to 3D indices and track legal moves.
3. Evaluate terminal states and use minimax with alpha-beta pruning.
4. Expose first-player and search-depth controls through command-line arguments.
5. Keep focused test files for both 2D and 3D behavior.

## Tech Stack

| Technology | Role |
|---|---|
| Python | Game logic and command-line execution |
| NumPy | 2D and 3D board representation |
| Colorama | Terminal color utilities |
| argparse | Player and search-depth options |

## Results and Deliverables

- Implemented a 3D board engine with minimax and alpha-beta pruning.
- Added configurable first-player and ply-depth command-line options.
- Included 2D and 3D test modules for focused behavioral checks.
- The 2D file remains a learning scaffold with unfinished win-evaluation logic; the 3D module is the primary implementation.

## Repository Contents

| Path | Purpose |
|---|---|
| `tictac3d.py` | 3D game engine, search logic, and CLI |
| `tictac.py` | Introductory 2D scaffold |
| `.tictac3d_test_*.py` | Focused 3D test cases |
| `.tictac_test_*.py` | Focused 2D test cases |

## Getting Started

Clone the repository:

```bash
git clone https://github.com/CS-Ponkoj/Tic_Tac_Toe_3D.git
cd Tic_Tac_Toe_3D
```

### Requirements

```bash
python -m pip install numpy colorama
```

### Run or Review

```bash
python tictac3d.py --player -1 --ply 6
```

## Reproducibility Notes

- Results above come from code, saved notebook outputs, or artifacts currently stored in this repository.
- Paths from the original development environment may need to be changed to repository-relative paths.
- Re-run the work after dependency changes before comparing new outputs with the recorded values.

## Limitations and Next Steps

- Search cost grows quickly as ply depth increases.
- The repository is an educational implementation, not a polished graphical game.
- Complete the 2D scaffold and consolidate the hidden test modules into a standard test suite.

## Author

**Ponkoj Shill**  
AI/ML researcher and Ph.D. candidate in Computer Science

- [GitHub](https://github.com/CS-Ponkoj)
- [Portfolio](https://ponkoj.com)

## License

No license file is currently included. Please contact the author before reusing the project beyond review, education, or fair-use evaluation.
