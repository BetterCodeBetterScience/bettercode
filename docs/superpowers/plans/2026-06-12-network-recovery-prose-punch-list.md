# Network-recovery prose punch list

Code changes for the validation chapter's "Simulating data from a model"
subsection landed in this branch (`latex-integration`). The chapter now does
partial-correlation recovery on an undirected skeleton of the ecoli70 graph
(no causal discovery, no timeseries, no `tigramite`, no `PCMCI`). The
following narrative needs author revision — the LLM did not edit any prose
per the `latex/CLAUDE.md` rule.

## Sentences and figure references that are now inaccurate

In `latex/book-validation.tex`, `\subsection{Simulating data from a model}`:

1. The `\textbf{TODO}` line at the top of the subsection can be removed.

2. **"Let's say that we have developed a tool that implements a novel method
   for the discovery of causal relationships from timeseries data."** — no
   longer about causal discovery or timeseries. Suggest: "Let's say we have
   developed a tool that recovers an undirected network from observational
   data."

3. **"generate data from a known causal graph (which is represented as a
   directed acyclic graph, just like our workflow graphs in the previous
   chapter)"** — now undirected. Suggest: "generate data from a known
   undirected network (the skeleton of a Bayesian network used in the gene
   expression literature)."

4. **"which has 46 nodes (representing individual genes) and 70 directed
   edges (representing causal relationships on gene expression between
   nodes)"** — edges are now treated as undirected. Suggest: "which has 46
   nodes (representing individual genes) and 70 edges (representing direct
   conditional dependencies between gene expression levels)."

5. **"Given this DAG, we then need to generate timeseries data for expression
   of each gene that reflect the causal relationships between the genes as
   well as the autocorrelation in gene expression within genes measured over
   time. For this, we turn to the *tigramite* package, which is primarily
   focused on causal discovery from timeseries data, but also includes a
   function that can generate timeseries data given a graphical model.
   However, the *tigramite* package requires a different representation of
   the graphical model than the one obtained from *pgmpy*, so we have to
   convert the edge representation from the original to the link format
   required for *tigramite*"** — entire paragraph needs replacement.
   Suggest: brief description of encoding the edge set as the sparsity
   pattern of a Gaussian precision matrix, with diagonal dominance for
   positive-definiteness, then sampling MVN data.

6. **"We can then create a function to take in the original model, convert
   it, and generate timeseries data for the model:"** — no longer
   timeseries. Suggest: "We then sample data from the multivariate normal
   distribution with this covariance:".

7. **"Now that we have the dataset we can test out our estimation method.
   Since I don't actually have a new method for causal estimation on
   timeseries, I will instead use the PCMCI method described by
   \citep{Runge:2019aa} and implemented in the *tigramite* package:"** —
   no longer using PCMCI or tigramite. Suggest: "Now that we have the
   dataset we can test out our estimation method, which estimates partial
   correlations between every pair of variables and thresholds their
   magnitudes to declare edges."

8. **"The results from this analysis include a list of all of the edges
   that were identified from the data using causal discovery, which we can
   summarize to determine how well the model performed. First we need to
   extract the links that were discovered from the results which pass our
   intended false discovery rate threshold:"** — `discover_edges` returns
   the set directly; there is no separate `extract_discovered_links` step
   and no q-value. The corresponding code listing has been DELETED; this
   paragraph now sits awkwardly without a following code block and should
   be deleted entirely.

   **Also delete the orphaned single-sentence paragraph "Then we can
   summarize the results:"** at line 212. After the previous paragraph
   and its code block are removed, this sentence is the lead-in to the
   `score_recovery` listing and reads naturally there — but if you choose
   to delete the preceding "First we need to extract…" paragraph, also
   delete this lead-in so the flow reads: prose about applying the
   estimation method → `discover_edges` listing → `score_recovery`
   listing → shell-output block.

9. **"The results showed that the model performed quite well, detecting all
   of the true relationships and only two false relationships."** — the
   new shell-output block reports 92 discovered edges with 25% FDR. Update
   the prose to match (e.g., "detecting nearly all of the true relationships
   while flagging about a quarter of the discovered edges as false
   positives" or similar).

10. **"For example, we would expect better model performance with stronger
    signal, and we would expect fewer nodes identified when the p-value
    threshold is more stringent."** — no p-value threshold any more; it's
    a partial-correlation magnitude threshold. Suggest: "…fewer edges
    identified when the threshold is more stringent."

11. **"The results confirm that the model is performing as expected, with
    increasing recall as a function of increasing true signal and
    decreasing FDR threshold."** — threshold acts on partial-correlation
    magnitude now, not FDR. Suggest: "…with increasing recall as a function
    of increasing true signal, and decreasing FDR as the partial-correlation
    threshold tightens."

12. **Figure caption (`\caption[Plot of true/false discovery rate as FDR
    increases]{A plot of observed true positive rate (TPR) and false
    discovery rate (FDR) at increasing signal levels for varying FDR
    thresholds.}`)** — the threshold is no longer an FDR threshold.
    Suggest: "A plot of observed recall and false discovery rate (FDR) at
    increasing signal levels and varying partial-correlation thresholds."

13. **`\includegraphics{files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png}`**
    must be updated to reference the new figure. The new figure is at
    `book/book/images/network_recovery_performance.png` (source location);
    the MyST/latex build rehashes filenames into `latex/files/`. Update the
    `\includegraphics` argument to whatever the new build produces (e.g.
    `files/network_recovery_performance-<newhash>.png`).

14. **Stale figure file at `latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png`** —
    this file is currently untracked (build-cache directory) and was not
    deleted by the LLM. After updating the `\includegraphics` line (item
    13) and rebuilding, the new hashed filename will appear and the old
    file can be deleted by the author when convenient. The LLM left it in
    place because deleting it before the prose/`\includegraphics` is
    updated could leave the next intermediate build broken.

## References that may become orphaned

- `\citep{Runge:2019aa}` no longer appears in the validation subsection.
  Check whether it is cited anywhere else in the book; if not, the
  bibliography entry can be removed from `latex/references.bib`.
- `\citep{Schafer:2005aa}` is still relevant (the graph is from there).

## Dependencies that may become orphaned

- `tigramite>=5.2.9.4` in `pyproject.toml` is no longer used in this
  chapter. Confirm it's not used elsewhere before removing.
- The obsolete `## graphical modeling` section in
  `notebooks/simulation_examples.py` is now redundant with the new standalone
  `notebooks/network_recovery_simulation.py`. The LLM did not delete the
  obsolete section because re-running that notebook (PyMC + clustering) is
  slow; the author can delete it manually at their convenience.
