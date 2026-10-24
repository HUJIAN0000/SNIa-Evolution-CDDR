# Changes made for this code/data release

The scientific input bytes, frozen posterior arrays, matching outputs and baseline priors were preserved. The release does not modify the source manuscript directories.

The portable package makes four small changes to existing script input/output handling:

1. `matching.py` reads the local `data/` directory and verifies/restores compressed inputs before loading them, rather than falling back to an author-specific parent folder.
2. `recalculate_baseline_mcmc.py` performs the same input preparation and creates its audit directory before the optional long baseline run.
3. `make_review_figures.py` writes figures to `figures/` without also requiring an unrelated manuscript directory.
4. `plot_baseline.py` is the prior `plot_results.py` recipe, reading the frozen `baseline/` layer and using installed PyMuPDF for PNG rendering instead of an external Poppler executable. Its posterior inputs, plotting helper, prior ranges and residual calculation are preserved.

The exact source diffs are in `packaging_io_changes.diff`. New entry-point, decompression, historical-replay and verification scripts handle packaging and execution only. They do not change the matching objective, GP equations, likelihood, error prescription or parameter priors.

The paper and response generators are left in the manuscript workspace because numerical reproduction does not depend on them. This release therefore does not propagate the recently identified prose-level AIC/BIC, tension or public-availability claims into a regenerated paper.
